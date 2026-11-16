void make_subset() {
    TFile input(
        "/vols/sbn/uboone/darkTridents/data/CNN_training/MPID_training_set_full.root",
        "READ"
    );

    TTree *tree = nullptr;
    input.GetObject("image2d_image2d_binary_tree", tree);

    if (!tree) {
        std::cerr << "Could not find input tree" << std::endl;
        return;
    }

    const Long64_t nentries = tree->GetEntries();
    const Long64_t ncopy = std::min<Long64_t>(1000, nentries);

    TFile output(
        "/home/hep/ma2922/Project/DM-CNN/test_data/MPID_training_subset_1000.root",
        "RECREATE"
    );

    TTree *subset = tree->CloneTree(ncopy);
    subset->Write();

    std::cout << "Copied " << ncopy << " of "
              << nentries << " entries" << std::endl;

    output.Close();
    input.Close();
}
