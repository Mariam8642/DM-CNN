from __future__ import division
from __future__ import print_function

import torch
print("Checking if CUDA is available: ")
print(torch.cuda.is_available())
print("\n")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import os, sys
from lib.config import config_loader
from lib.utility import timestr
import numpy as np
import time

import optuna

# import MPID core
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary, mpid_func_binary

# Config path is module level so both TrainCNN and objective can use it
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CFG = os.path.join(BASE_PATH, "../cfg", "training_config_smoketest.cfg")


def TrainCNN(cfg, trial=None):
    '''
    Trains the DM-CNN model using the settings in cfg.
    If an Optuna trial is given, intermediate test losses are reported
    to it so bad trials can be pruned.

    Returns the best test loss seen during training.
    '''

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUID

    title = cfg.name
    output_dir = cfg.output_directory
    train_file = cfg.input_train
    test_file = cfg.input_test
    weights_dir = cfg.weights_directory
    SEED = cfg.seed_number

    print("Inputs given: ")
    print("Training file: ", train_file)
    print("Test file: ", test_file)
    print("Output directory: ", output_dir)
    print("Weights directory: ", weights_dir)
    print("Seed?: ", cfg.seed)
    print("\n")

    fout = open(output_dir + 'DM-CNN_training_metrics_{}_{}.csv'.format(timestr(), title), 'w')
    fout.write('train_accu,test_accu,train_loss,test_loss,epoch,step')
    fout.write('\n')

    CNN_weights = weights_dir + "DM-CNN_model_{}_epoch_{}_batch_id_{}_labels_{}_title_{}_step_{}.pwf"

    cuda = torch.cuda.is_available()

    # For reproducibility (manual_seed seeds CPU and CUDA)
    if cfg.seed:
        print("Using seed number: ", SEED)
        torch.manual_seed(SEED)
        if cuda:
            torch.cuda.manual_seed(SEED)
    else:
        print("A seed has not been defined...\n")

    print("There are {} GPUs available".format(torch.cuda.device_count()))
    train_device = 'cuda' if cuda else 'cpu'

    # Training data
    train_data = mpid_data_binary.MPID_Dataset(train_file, "image2d_image2d_binary_tree", train_device, plane=0, augment=False)
    train_data = Subset(train_data, range(min(500, len(train_data))))
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size_train, shuffle=True)
    labels = 2

    # Test data
    test_data = mpid_data_binary.MPID_Dataset(test_file, "image2d_image2d_binary_tree", train_device, plane=0)
    test_data = Subset(test_data, range(min(100, len(test_data))))
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size_test, shuffle=True)

    mpid = mpid_net_binary.MPID(dropout=cfg.drop_out, num_classes=2)
    mpid = mpid.to(train_device)

    loss_fn = nn.BCEWithLogitsLoss()

    weight_decay = getattr(cfg, "weight_decay", 0.0)
    optimizer = optim.Adam(mpid.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay)
    train_step = mpid_func_binary.make_train_step(mpid, loss_fn, optimizer)
    test_step = mpid_func_binary.make_test_step(mpid, test_loader, loss_fn, optimizer)

    print("Training with {} images".format(len(train_loader.dataset)))

    EPOCHS = cfg.EPOCHS
    print("Start DM-CNN training...")

    step = 0
    best_test_loss = float("inf")
    last_loss = None

    init = time.time()
    try:
        for epoch in range(EPOCHS):
            print("\n")
            print(" @{}th epoch...".format(epoch))
            for batch_idx, (x_batch, y_batch, info_batch, nevents_batch) in enumerate(train_loader):
                x_batch = x_batch.to(train_device).view((-1, 1, 512, 512))
                y_batch = y_batch.to(train_device)
                loss = train_step(x_batch, y_batch)  # model.train() called in train_step
                last_loss = loss
                print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, EPOCHS - 1,
                    batch_idx * len(x_batch), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss),
                    end='')

                if (batch_idx % cfg.test_every_step == 1 and cfg.run_test):
                    if cfg.save_weights:
                        torch.save(mpid.state_dict(), CNN_weights.format(timestr(), epoch, batch_idx, labels, title, step))

                    print("Start eval on test sample.......@step..{}..@epoch..{}..@batch..{}".format(step, epoch, batch_idx))
                    test_accuracy = mpid_func_binary.validation(mpid, test_loader, cfg.batch_size_test, train_device, event_nums=cfg.test_events_nums)
                    print("Test Accuracy {}".format(test_accuracy))
                    print("Start eval on training sample...@epoch..{}.@batch..{}".format(epoch, batch_idx))
                    train_accuracy = mpid_func_binary.validation(mpid, train_loader, cfg.batch_size_train, train_device, event_nums=cfg.test_events_nums)
                    print("Train Accuracy {}".format(train_accuracy))
                    test_loss = test_step(test_loader, train_device)
                    print("Test Loss {}".format(test_loss))

                    fout.write("%f," % train_accuracy)
                    fout.write("%f," % test_accuracy)
                    fout.write("%f," % loss)
                    fout.write("%f," % test_loss)
                    fout.write("%d," % epoch)
                    fout.write("%d" % step)
                    fout.write("\n")
                    fout.flush()

                    best_test_loss = min(best_test_loss, test_loss)

                    if trial is not None:
                        trial.report(test_loss, step)  # step is unique within a trial
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                step += 1

        # If the eval condition never fired (small subset), evaluate once at the end
        # so the trial never returns inf.
        if best_test_loss == float("inf"):
            best_test_loss = test_step(test_loader, train_device)
            print("Final Test Loss {}".format(best_test_loss))
    finally:
        fout.close()

    print("\n")
    print("Total training time: {:0.4f} seconds".format(time.time() - init))
    return best_test_loss


def objective(trial):
    cfg = config_loader(CFG)

    cfg.learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg.drop_out = trial.suggest_float("dropout", 0.0, 0.5)
    cfg.batch_size_train = trial.suggest_categorical("batch_size", [16, 32, 64])
    # cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    cfg.save_weights = False                                   # no .pwf file per trial
    cfg.name = "{}_trial{}".format(cfg.name, trial.number)     # separate CSV per trial

    try:
        return TrainCNN(cfg, trial)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()   # big batch sizes won't crash the whole study


if __name__ == '__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    study = optuna.create_study(
        direction="minimize",
        study_name="dmcnn_smoketest",
        storage="sqlite:///dmcnn_optuna.db",   # resume later / run several workers
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
