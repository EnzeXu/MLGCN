class Config:
    main_path = ""
    dataset = "GCN_N3P"
    max_natoms = 126
    length = 1600
    train_length = 1000
    test_length = length - train_length
    root_bmat = main_path + 'data1/GCN_N3P/BTMATRIXES/'
    root_dmat = main_path + 'data1/GCN_N3P/DMATRIXES/'
    root_conf = main_path + 'data1/GCN_N3P/CONFIGS/'
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "COORD_{}"


config = Config

