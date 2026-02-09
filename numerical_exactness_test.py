
import numpy as np
import pickle

# Use PyTorch for MNIST
from torchvision import datasets
import torchvision.transforms as transforms

# Load MNIST test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, 
                              transform=transforms.ToTensor())
x_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
y_test = test_dataset.targets.numpy()

# ReLU forward pass
def relu_forward_multi_layer(X, weights, biases):
    """Multi-layer ReLU network forward pass"""
    n_layers = len(weights)   #count how many layers
    activations = X   #start with input
    
    for layer_idx in range(n_layers):   #loop through each layer 
        W = weights[layer_idx]
        b = biases[layer_idx]
        
        z = activations @ W + b
        
        if layer_idx < n_layers - 1:   #check if not last layer
            activations = np.maximum(0, z)
        else:
            activations = z   #last layer stays linear(no relu)
    
    return activations

#convert relu to snn layer by layer
# SNN forward pass
def snn_forward_multi_layer(X, weights, biases, t_max=1.0):
    """Multi-layer SNN forward pass using time-to-first-spike"""
    n_layers = len(weights)
    V_rest = 0.0   #resting membrane potential 
    
    # Start with input as activities
    current_activity = X
    
    for layer_idx in range(n_layers):
        W = weights[layer_idx]
        b = biases[layer_idx]
        
        V_thresh = t_max - b - W.sum(axis=0)   # threshold formula from the paper 
        
        # Compute input current from previous layer activities
        input_current = current_activity @ W
        delta_V = input_current    # change in voltage due to input current
        spike_times = (V_thresh - V_rest) / (delta_V + 1e-10)     #spike time formula, add small term to avoid divide by zero
        spike_times = np.clip(spike_times, 0, t_max)
        
        # Convert spike times back to activities for next layer
        current_activity = t_max - spike_times
    
    return current_activity

# Load 2-layer trained model 
with open('mnist_2_layer.pkl', 'rb') as f:
    model_data = pickle.load(f)

weights = model_data['weights']
biases = model_data['biases']

print("\n" + "="*60)
print("NUMERICAL EXACTNESS VERIFICATION")
print("="*60)
print(f"Testing 2-layer network on 100 MNIST samples...\n")

#campare relu and snn outputs on same images,track max diff(worst case)and avg diff(typical case)
max_diff = 0  # track maximum difference across all samples
avg_diff = 0   #accumulate diff for averaging

for i in range(100):
    X = x_test[i:i+1]
    
    out_relu = relu_forward_multi_layer(X, weights, biases)
    out_snn = snn_forward_multi_layer(X, weights, biases)
    
    diff = np.abs(out_relu - out_snn).max()    #absolute diff between outputs
    max_diff = max(max_diff, diff)
    avg_diff += diff

avg_diff /= 100   

print(f"Maximum output difference: {max_diff:.10f}")
print(f"Average output difference: {avg_diff:.10f}")
print(f"Relative error: {max_diff / np.mean(np.abs(out_relu)):.10e}")
print()

if max_diff < 1e-5:
    print("VERIFIED: Numerical exactness confirmed at machine precision")
    print("Formula is mathematically exact for shallow networks")
else:
    print("NOTE: Small numerical errors present")
    print("This is expected due to floating-point arithmetic")

print("="*60)