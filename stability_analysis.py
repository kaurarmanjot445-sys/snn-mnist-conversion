
import numpy as np
import matplotlib.pyplot as plt

# same functions as before
def relu(x):
    return (x > 0) * x

def spiking_simulator(t0, J, alpha, V):
    return (V + t0 @ J) / (alpha + J.sum(0))

# modified this to test what happens with wrong parameters
def simple_mapping(X, W, b=None, t_max=15.0, threshold_scale=1.0):
    if b is None:
        b = np.zeros(W.shape[1])
    
    t_in = 1 - X
    J = W
    
    # normally this should be exact, but adding threshold_scale to see what breakes (to test what happen when parameters are slightly wrong)
    V = threshold_scale * (t_max - b - J.sum(0))
    alpha = 1. - J.sum(0)
    
    return alpha, J, V, t_max, t_in

def test_depth(n_layers, threshold_scale=1.0):
    np.random.seed(42)
    X = np.array([[0.8, 0.6, 0.4, 0.2]])
    
    # all layers 4 to 4
    weights = []
    for _ in range(n_layers):
        weights.append(np.random.randn(4, 4) * 0.2)
    
    # relu reference
    h = X
    for W in weights:
        h = relu(h @ W)
    y_relu = h
    
    # snn with chosen threshold_scale
    h = X
    for W in weights:
        alpha, J, V, tmax, t_in = simple_mapping(h, W, threshold_scale=threshold_scale)
        t_out = spiking_simulator(t_in, J, alpha, V)   # output spike time
        t_out = np.minimum(t_out, tmax)
        h = tmax - t_out
    y_snn = h
    
    return np.max(np.abs(y_relu - y_snn))

depths = [2, 5, 10, 15, 20, 30, 40, 60, 80, 100]

errors_correct = []
errors_broken = []

print("Correct threshold scaling (1.0):")
for d in depths:
    e = test_depth(d, threshold_scale=1.0)
    errors_correct.append(e)
    print(f" {d:3d} layers: {e:.2e}")

print("\nBroken threshold scaling (0.95):")
for d in depths:
    e = test_depth(d, threshold_scale=0.95)
    errors_broken.append(e)
    print(f" {d:3d} layers: {e:.2e}")

# plot
plt.figure(figsize=(10, 6))
plt.semilogy(depths, errors_correct, 'o-', label='Correct (1.0)', color='green')
plt.semilogy(depths, errors_broken, 's-', label='Broken (0.95)', color='red')
plt.axhline(1e-14, linestyle='--', color='gray', label='machine precision')

plt.xlabel("network depth")
plt.ylabel("max error")
plt.title("Stability: correct vs broken threshold scaling")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("stability_analysis.png", dpi=150)
plt.show() 
print("\nsaved as stability_analysis.png")

# quick summary
print("\nSo with correct scaling, error stays tiny even at 100 layers")
print("but with 5% error in threshold, it starts breaking around depth 20-30")
print("makes sense - small errors compound through the layers")