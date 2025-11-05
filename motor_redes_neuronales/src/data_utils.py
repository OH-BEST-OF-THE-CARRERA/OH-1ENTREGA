import numpy as np

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, shuffle=True):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1"
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    X = X[indices]
    y = y[indices]

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_mini_batches(X, y, batch_size):
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]
