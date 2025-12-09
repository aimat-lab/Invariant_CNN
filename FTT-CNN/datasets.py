import numpy as np

def load_flow_dataset():
    # ~ train set
    x_train = np.load('./dataset/x_removed_island.npy')
    y_train = np.load('./dataset/y_removed_island.npy')
    
    # ~ test set
    x_test = np.load('./dataset/x_test_removed_island.npy')
    y_test = np.load('./dataset/y_test_removed_island.npy')
    
    # ~ flat channels
    x_flat = np.load('./dataset/x_flat.npy')
    y_flat = np.load('./dataset/y_flat.npy')
    
    return (
        np.concatenate([x_train, x_flat, x_test]),
        np.concatenate([y_train, y_flat, y_test]),
    )
