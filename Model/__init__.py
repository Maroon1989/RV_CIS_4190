from model import *
STRATEGIES = {
    'Lasso':build_lasso,
    'Xgboost':build_xgboost
    
}

# pytorch deep learning/ cnn 
DEEP = {
    'CNN':build_CNN,
    'NN': build_nn
}