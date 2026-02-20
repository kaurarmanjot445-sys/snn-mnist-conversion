
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)


def relu(x):
    return np.maximum(0, x)


def compute_t_max(x_relu, b, t_min):
    max_act = float(np.max(x_relu))
    max_b = float(np.max(np.abs(b)))
    return t_min + (max_act + max_b) + 0.01


def error_correct(n_layers, n=50, batch=30):
    # B1 mapping with principled t_max — should give machine precision
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (batch, n))
    Ws = [rng.normal(0, 0.15, (n, n)) for _ in range(n_layers)]
    bs = [np.zeros(n)] * n_layers

    acts = [X]
    h = X
    for W, b in zip(Ws, bs):
        h = relu(h @ W + b)
        acts.append(h)

    t_in, t_prev, max_err = 1.0 - X, 1.0, 0.0
    for i, (W, b) in enumerate(zip(Ws, bs)):
        t_min = t_prev
        t_max = compute_t_max(acts[i + 1], b, t_min)
        V = (t_max - t_min) - b
        t_out = np.clip(t_min + V - (t_min - t_in) @ W, t_min, t_max)
        x_snn = np.maximum(t_max - t_out, 0.0)
        max_err = max(max_err, float(np.max(np.abs(acts[i + 1] - x_snn))))
        t_in, t_prev = t_out, t_max
    return max_err


def error_wrong(n_layers, n=50, batch=30):
    # Same thing but t_max is fixed at 0.5 — too small, clips activations
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (batch, n))
    Ws = [rng.normal(0, 0.15, (n, n)) for _ in range(n_layers)]
    bs = [np.zeros(n)] * n_layers

    acts = [X]
    h = X
    for W, b in zip(Ws, bs):
        h = relu(h @ W + b)
        acts.append(h)

    t_in, t_prev, max_err = 1.0 - X, 1.0, 0.0
    for i, (W, b) in enumerate(zip(Ws, bs)):
        t_min = t_prev
        t_max = t_min + 0.5 # wrong: ignores the actual activation range
        V = (t_max - t_min) - b
        t_out = np.clip(t_min + V - (t_min - t_in) @ W, t_min, t_max)
        x_snn = np.maximum(t_max - t_out, 0.0)
        max_err = max(max_err, float(np.max(np.abs(acts[i + 1] - x_snn))))
        t_in, t_prev = t_out, t_max
    return max_err


layers = list(range(1, 13))
correct = [error_correct(L) for L in layers]
wrong = [error_wrong(L) for L in layers]

# Print the table
print(f"{'L':>4} | {'Correct t_max':>14} | {'Wrong t_max':>12}")
print("-" * 38)
for L, c, w in zip(layers, correct, wrong):
    print(f"{L:>4} | {c:>14.2e} | {w:>12.2e}")

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(layers, correct, 'o-', color='royalblue', linewidth=2.5,
        markersize=8, label='B1 mapping (correct t_max)')
ax.plot(layers, wrong, 's--', color='tomato', linewidth=2.5,
        markersize=8, label='Wrong t_max (fixed too small)')
ax.axhline(y=1e-4, color='gray', linestyle=':',
           linewidth=1.5, label='Tolerance 1e-4')

ax.set_yscale('log')
ax.set_xticks(layers)
ax.set_xlabel('Number of Hidden Layers', fontsize=12)
ax.set_ylabel('Max |ReLU - SNN| Error', fontsize=12)
ax.set_title('Why t_max matters: correct vs wrong\n'
             'Correct gives machine precision, wrong gives ~1.0 error',
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stability_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved stability_analysis.png")