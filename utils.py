import numpy as np

def preprocess_image(X, game='Pong-v0'):
    """
    Downsample and preprocess game frame, pong preprocessing taken from Karpathy blog post
    """
    if game == 'Pong-v0':
        X = X[35:195]
        X = X[::2, ::2, 0]
        X[X == 144] = 0
        X[X == 109] = 0
        X[X != 0] = 1
        return X.astype(np.float).ravel()

    return X