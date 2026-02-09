
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms

# Load MNIST test set
test_dataset = datasets.MNIST(root='./data', train=False, download=True, 
                              transform=transforms.ToTensor())
x_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0

# SNN forward pass that returns spike times
def snn_forward_with_spike_times(X, weights, biases, t_max=1.0):
    """Returns final output AND all layer spike times"""
    n_layers = len(weights)
    V_rest = 0.0
    all_spike_times = []    #empty list to collect spike times per layer
    current_activity = X
    
    #same snn pass,but save spike times after each layer
    for layer_idx in range(n_layers):
        W = weights[layer_idx]
        b = biases[layer_idx]
        
        V_thresh = t_max - b - W.sum(axis=0)
        input_current = current_activity @ W
        
        delta_V = input_current
        spike_times = (V_thresh - V_rest) / (delta_V + 1e-10)
        spike_times = np.clip(spike_times, 0, t_max)
        
        all_spike_times.append(spike_times.copy())   #save spike times for this layer
        current_activity = t_max - spike_times
    
    return all_spike_times

# Load models
models = {
    2: 'mnist_2_layer.pkl',
    4: 'mnist_4_layer.pkl',
    6: 'mnist_6_layer.pkl',
    8: 'mnist_8_layer.pkl'
}

spike_collections = {}

print("Collecting spike times from all models...")
for n_layers, filename in models.items():
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    weights = model_data['weights']
    biases = model_data['biases']
    
    # Collect spikes from 200 samples
    all_final_spikes = []
    for i in range(200):
        X = x_test[i:i+1]
        layer_spikes = snn_forward_with_spike_times(X, weights, biases)  # Get last layer spikes
        all_final_spikes.extend(layer_spikes[-1].flatten())
    
    #store collected spike times for this depth
    spike_collections[n_layers] = np.array(all_final_spikes)
    print(f"{n_layers}-layer: collected {len(all_final_spikes)} spike times")

# Create visualization(flatten to 1D)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

depths = [2, 4, 6, 8]
colors = ['blue', 'orange', 'green', 'red']

for idx, (n_layers, color) in enumerate(zip(depths, colors)):
    spikes = spike_collections[n_layers]     
    
    axes[idx].hist(spikes, bins=50, color=color, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{n_layers}-Layer Network', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Spike Time', fontsize=11)
    axes[idx].set_ylabel('Count', fontsize=11)
    axes[idx].axvline(np.mean(spikes), color='darkred', linestyle='--', 
                     linewidth=2, label=f'Mean: {np.mean(spikes):.3f}')
    axes[idx].axvline(1.0, color='gray', linestyle=':', linewidth=2, 
                     label='t_max = 1.0', alpha=0.7)
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    
    # highlight the clustering problem(key insight)
    if n_layers == 8:
        axes[idx].text(0.5, 0.95, 'Clustering near t_max', 
                      transform=axes[idx].transAxes, fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                      verticalalignment='top', horizontalalignment='center')

plt.suptitle('Spike Time Distributions: Depth-Dependent Clustering', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('spike_time_distributions.png', dpi=150, bbox_inches='tight')
print("\nSaved spike_time_distributions.png")
plt.show()