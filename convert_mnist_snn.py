

import numpy as np
import pickle

# Load MNIST
def load_mnist():
    from tensorflow import keras
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return x_test, y_test

# Load trained weights ,load saved weights from training script
with open('mnist_weights.pkl', 'rb') as f:
    weights = pickle.load(f)

W1, b1 = weights['W1'], weights['b1']
W2, b2 = weights['W2'], weights['b2']
W3, b3 = weights['W3'], weights['b3']

print("Weights loaded")
print(f"W1: {W1.shape}, W2: {W2.shape}, W3: {W3.shape}")

# SNN functions
def relu(x):
    return np.maximum(0, x)

def spiking_simulator(t0, J, alpha, V):      #use integrete-and-fire equation from ppr
    t1 = (V + t0 @ J) / (alpha + J.sum(0))   # the equation
    return t1

def simple_mapping(X, W, b, t_max=15.0):
    X_min, X_max = np.min(X), np.max(X)
    if X_max - X_min > 1e-8:
        X_norm = (X - X_min) / (X_max - X_min)
        scale = X_max - X_min
    else:
        X_norm = X
        scale = 1.0
    
    t_in = 1 - X_norm  #encode as spike times
    J = W
    V = t_max - b - J.sum(0)
    alpha = 1.0 - J.sum(0)
    
    return alpha, J, V, t_max, t_in, scale

def snn_forward(X, W1, b1, W2, b2, W3, b3):
    alpha1, J1, V1, tmax1, tin1, scale1 = simple_mapping(X, W1, b1)
    tout1 = spiking_simulator(tin1, J1, alpha1, V1)
    tout1 = np.minimum(tout1, tmax1)
    h1 = (tmax1 - tout1) * scale1
    
    alpha2, J2, V2, tmax2, tin2, scale2 = simple_mapping(h1, W2, b2)
    tout2 = spiking_simulator(tin2, J2, alpha2, V2)
    tout2 = np.minimum(tout2, tmax2)
    h2 = (tmax2 - tout2) * scale2
    
    alpha3, J3, V3, tmax3, tin3, scale3 = simple_mapping(h2, W3, b3)
    tout3 = spiking_simulator(tin3, J3, alpha3, V3)
    tout3 = np.minimum(tout3, tmax3)
    output = (tmax3 - tout3) * scale3
    
    return output

# Load test data
X_test, y_test = load_mnist()

#Test on FULL 10,000 samples, we check if conersion works on all test images not just 100
print("\nTesting SNN conversion on FULL TEST SET (10,000 samples)...")
print("This will take a few minutes...\n")

n_test = 10000 # FULL TEST SET

correct_relu = 0
correct_snn = 0
errors = []

for i in range(n_test):
    X = X_test[i:i+1]
    y_true = y_test[i]
    
    # ReLU prediction
    h1_relu = relu(X @ W1 + b1)
    h2_relu = relu(h1_relu @ W2 + b2)
    out_relu = h2_relu @ W3 + b3
    pred_relu = np.argmax(out_relu)
    
    # SNN prediction
    out_snn = snn_forward(X, W1, b1, W2, b2, W3, b3)
    pred_snn = np.argmax(out_snn)
    
    # Track error
    error = np.mean(np.abs(out_relu - out_snn))
    errors.append(error)
    
    if pred_relu == y_true:
        correct_relu += 1
    if pred_snn == y_true:
        correct_snn += 1
    
    # Progress updates
    if (i+1) % 1000 == 0:
        print(f"Tested {i+1}/{n_test} samples...")

# Calculate metrics
acc_relu = 100.0 * correct_relu / n_test
acc_snn = 100.0 * correct_snn / n_test
mean_error = np.mean(errors)
max_error = np.max(errors)

# DETAILED RESULTS
print("\n" + "="*60)
print("FULL MNIST TEST SET RESULTS")
print("="*60)
print(f"Total samples: {n_test}")
print(f"\nReLU Network:")
print(f" Correct predictions: {correct_relu}/{n_test}")
print(f" Accuracy: {acc_relu:.2f}%")
print(f"\nSNN Network:")
print(f" Correct predictions: {correct_snn}/{n_test}")
print(f" Accuracy: {acc_snn:.2f}%")
print(f"\nConversion Quality:")
print(f" Accuracy drop: {acc_relu - acc_snn:.2f}%")
print(f" Mean output error: {mean_error:.4f}")
print(f" Max output error: {max_error:.4f}")

# Analysis
if abs(acc_relu - acc_snn) < 1.0:
    print(f"\n✓ Excellent: SNN maintains ReLU accuracy (< 1% drop)")
elif abs(acc_relu - acc_snn) < 3.0:
    print(f"\n~ Good: Minor accuracy degradation (< 3% drop)")
else:
    print(f"\n✗ Significant accuracy loss (> 3% drop)")

print("="*60)