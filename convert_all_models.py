import numpy as np
import pickle
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

SAVE_DIR = './saved_models'
LAYER_COUNTS = [2, 4, 6, 8]


def relu(x):
    return np.maximum(0, x)


def compute_t_max(x_relu, b, t_min):
    # We need t_max large enough that every active neuron can fire,before the window closes. Derived from Eq. 9 of the paper.
    max_act = float(np.max(x_relu))
    max_b = float(np.max(np.abs(b)))
    return t_min + (max_act + max_b) + 0.01# The +0.01 is just a tiny float safety margin.


def relu_forward(X, weights, biases):
    # Returns activations at each layer so WE can use them for t_max.
    acts = [X]
    h = X
    for W, b in zip(weights[:-1], biases[:-1]):
        h = relu(h @ W + b)
        acts.append(h)
    acts.append(h @ weights[-1] + biases[-1])
    return acts


def snn_forward(X, weights, biases):
    # SNN forward pass using the B1 identity mapping.

    relu_acts = relu_forward(X, weights, biases)

    t_in = 1.0 - X # high pixel value -> early spike
    t_prev = 1.0
    snn_acts = [X]
    sparsity = []

    for n, (W, b) in enumerate(zip(weights[:-1], biases[:-1])):
        t_min = t_prev
        t_max = compute_t_max(relu_acts[n + 1], b, t_min)

        V = (t_max - t_min) - b # threshold (Eq. 9)
        delta = t_min - t_in
        t_out = np.clip(t_min + V - delta @ W, t_min, t_max)
        x_out = np.maximum(t_max - t_out, 0.0) # convert spike time back to activation

        sparsity.append(float((t_out < t_max - 1e-9).mean()))
        snn_acts.append(x_out)
        t_in = t_out
        t_prev = t_max

    snn_acts.append(snn_acts[-1] @ weights[-1] + biases[-1])
    return snn_acts, sparsity


def load_mnist(n=10000):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    ds = torchvision.datasets.MNIST('./data', train=False,
                                     download=True, transform=tf)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    X, y = next(iter(loader))
    return X.numpy()[:n], y.numpy()[:n]


if __name__ == '__main__':
    print("Loading MNIST test set (10000 samples)...")
    X, y = load_mnist()
    X_verify = X[:200] # small batch just for the error check

    results = {}

    print("\n" + "="*65)
    print(f" {'L':>3} | {'ReLU':>8} | {'SNN':>8} | "
          f"{'Drop':>7} | {'Sparsity':>9} | Layer errors")
    print(" " + "-"*65)

    for L in LAYER_COUNTS:
        path = os.path.join(SAVE_DIR, f'model_L{L}.pkl')
        if not os.path.exists(path):
            print(f" L={L}: file not found, run train_mnist_pytorch.py first.")
            continue

        with open(path, 'rb') as f:
            data = pickle.load(f)
        W, b = data['weights'], data['biases']

        relu_acts = relu_forward(X, W, b)
        relu_acc = float(np.mean(np.argmax(relu_acts[-1], 1) == y))

        snn_acts, spl = snn_forward(X, W, b)
        snn_acc = float(np.mean(np.argmax(snn_acts[-1], 1) == y))
        sparsity = float(np.mean(spl))
        drop = relu_acc - snn_acc

        # Check how close SNN and ReLU activations are at each hidden layer
        relu_v = relu_forward(X_verify, W, b)
        snn_v, _ = snn_forward(X_verify, W, b)
        errs = [float(np.max(np.abs(relu_v[n] - snn_v[n])))
                    for n in range(1, len(W))]
        err_str = " ".join(f"L{n+1}={e:.0e}" for n, e in enumerate(errs))

        print(f" {L:>3} | {relu_acc*100:>7.2f}% | {snn_acc*100:>7.2f}% | "
              f"{drop*100:>6.3f}% | {sparsity:>9.4f} | {err_str}")

        results[L] = {
            'relu_acc': relu_acc, 'snn_acc': snn_acc,
            'drop': drop, 'sparsity': sparsity, 'spl': spl,
        }

    print("="*65)
    if results:
        with open('conversion_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("\nSaved results to conversion_results.pkl")
        print("Run create_plot.py next.")
    else:
        print("\nNo results saved. Run train_mnist_pytorch.py first.")