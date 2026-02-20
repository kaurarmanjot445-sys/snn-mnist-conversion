
import numpy as np

np.random.seed(42)


def relu(x):
    return np.maximum(0, x)


def compute_t_max(x_relu, b, t_min):
    # Need t_max big enough that every active neuron fires inside the window.
    max_act = float(np.max(x_relu))
    max_b = float(np.max(np.abs(b)))
    return t_min + (max_act + max_b) + 0.01


def verify(weights, biases, X, label):
    # The t_min of each layer equals the t_max of the previous one.

    # ReLU forward
    relu_acts = [X]
    h = X
    for W, b in zip(weights, biases):
        h = relu(h @ W + b)
        relu_acts.append(h)

    # SNN forward, checking each layer as we go
    t_in = 1.0 - X # encode input as spike times
    t_prev = 1.0
    all_pass = True

    print(f"\n{label}")
    print(f" {'Layer':>6} | {'t_min':>7} | {'t_max':>7} | {'Max Error':>12} | Status")
    print(" " + "-"*52)

    for n, (W, b) in enumerate(zip(weights, biases)):
        t_min = t_prev
        t_max = compute_t_max(relu_acts[n + 1], b, t_min)
        V = (t_max - t_min) - b
        t_out = np.clip(t_min + V - (t_min - t_in) @ W, t_min, t_max)
        x_snn = np.maximum(t_max - t_out, 0.0)

        err = float(np.max(np.abs(relu_acts[n + 1] - x_snn)))
        passed = err < 1e-9
        if not passed:
            all_pass = False

        print(f" {n+1:>6} | {t_min:>7.4f} | {t_max:>7.4f} | "
              f"{err:>12.2e} | {'PASS' if passed else 'FAIL'}")

        t_in = t_out
        t_prev = t_max # chain: next layer's t_min = this layer's t_max

    print(f" => {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == '__main__':
    rng = np.random.default_rng(42)

    # Single layer first
    X = rng.uniform(0, 1, (4, 6))
    W = rng.normal(0, 0.3, (6, 4))
    b = rng.normal(0, 0.1, (4,))
    p1 = verify([W], [b], X, "Single layer (4 patterns, 6 -> 4 neurons)")

    # Then extend to 3 layers to make sure chaining works
    X2 = rng.uniform(0, 1, (8, 10))
    Ws = [rng.normal(0, 0.2, (10, 8)),
          rng.normal(0, 0.2, (8, 6)),
          rng.normal(0, 0.2, (6, 4))]
    bs = [rng.normal(0, 0.05, (8,)),
          rng.normal(0, 0.05, (6,)),
          rng.normal(0, 0.05, (4,))]
    p2 = verify(Ws, bs, X2, "3-layer network (8 patterns, 10->8->6->4)")

    print(f"\n{'All checks passed. Safe to proceed.' if p1 and p2 else 'Something is wrong, check the math.'}")