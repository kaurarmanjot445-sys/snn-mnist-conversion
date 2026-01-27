
import numpy as np
import matplotlib.pyplot as plt

# took these directly from the notebook
def relu(x):     #paper shows exact equivalanve betw relu and SNN
    return (x > 0) * x #positive stays,negative zero

def spiking_simulator(t0, J, alpha, V):  #output neurons spike based on input spike times
    t1 = (V + t0 @ J) / (alpha + J.sum(0)) #core SNN equation, calculates when the output spikes
    
    for i in range(t0.shape[0]):
        if np.min(t1[i]) < np.max(t0[i]):  #derived from LIF 
            print(f"Warning: timing issue in sample {i}")
    
    return t1

def simple_mapping(X, W, b=None):  #convert relu to SNN
    if b is None:
        b = np.zeros(W.shape[1])
    
    t_min_prev, t_max_prev = 0, 1
    t_in = 1 - X   #spike times
    t_min = t_max_prev
    J = W
    t_max = 15.
    V = t_max - b - J.sum(0) #critical formula from the paper 
    alpha = 1. - J.sum(0) # maintains mathematical equivalence with ReLU
    
    return alpha, J, V, t_min, t_max, t_in

# testing 2 layers
print("\n2-layer test\n")

X = np.array([[0.8, 0.5, 0.3, 0.1]])

# layer 1: 4 to 3
W1 = np.array([
    [0.5, -0.3, 0.2],
    [0.3, 0.4, -0.1],
    [0.1, 0.2, 0.3],
    [-0.2, 0.1, 0.4]
])

# layer 2: 3to→ 2
W2 = np.array([
    [0.6, -0.2],
    [0.3, 0.4],
    [-0.1, 0.5]
])

print(f"W1 shape: {W1.shape}")
print(f"W2 shape: {W2.shape}")

# ReLU version
h1_relu = relu(X @ W1)
y_relu = relu(h1_relu @ W2)   # standard ANN forward pass

print("\nReLU outputs:")
print("h1 =", h1_relu[0])
print("y =", y_relu[0])

# SNN version - layer by layer  , what are we testing 

# layer 1
alpha1, J1, V1, t_min1, t_max1, t_in1 = simple_mapping(X, W1)
t_out1 = spiking_simulator(t_in1, J1, alpha1, V1)
t_out1 = np.minimum(t_out1, t_max1)
h1_snn = t_max1 - t_out1 # key decoding step

# layer 2
alpha2, J2, V2, t_min2, t_max2, t_in2 = simple_mapping(h1_snn, W2)   #key test- can we chain layer? 
t_out2 = spiking_simulator(t_in2, J2, alpha2, V2)
t_out2 = np.minimum(t_out2, t_max2)
y_snn = t_max2 - t_out2 

print("\nSNN outputs:")
print("h1 =", h1_snn[0])
print("y =", y_snn[0])

error1 = np.abs(h1_relu - h1_snn)
err2 = np.abs(y_relu - y_snn) # inconsistent naming

print("\nErrors:")
print("Layer 1:", np.max(error1))
print("Layer 2:", np.max(err2))

if np.max(err2) < 1e-6:
    print("\nLooks exact")

# wanted to see how this scales ,testing deeper networks
def test_n_layers(X, n_layers):
    np.random.seed(42)
    
    weights = []
    for i in range(n_layers):
        if i == 0:
            W = np.random.randn(4, 3) * 0.3
        else:
            W = np.random.randn(3, 3) * 0.3
        weights.append(W)
    
    # relu
    h_relu = X
    for W in weights:
        h_relu = relu(h_relu @ W)
    
    # snn
    h_snn = X
    for W in weights:
        alpha, J, V, _, tmax, t_in = simple_mapping(h_snn, W)
        t_out = spiking_simulator(t_in, J, alpha, V)
        t_out = np.minimum(t_out, tmax)
        h_snn = tmax - t_out
    
    return np.max(np.abs(h_relu - h_snn))

print("\nTesting different depths:")
X_test = np.array([[0.8, 0.6, 0.4, 0.2]])

for n in [2, 3, 4, 5, 6, 8, 10]:
    e = test_n_layers(X_test, n)
    print(f"{n:2d} layers -> error = {e:.2e}")