import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('conversion_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract data(layer counts and accuracies into separate lists for plotting)
layers = sorted(results.keys())
relu_accs = [results[n]['relu_acc'] for n in layers]
snn_accs = [results[n]['snn_acc'] for n in layers]

# Create plot
plt.figure(figsize=(10, 6))

plt.plot(layers, relu_accs, 'o-', linewidth=2, markersize=8,
         label='ReLU Network', color='blue')
plt.plot(layers, snn_accs, 's-', linewidth=2, markersize=8,
         label='SNN Conversion', color='red')

plt.xlabel('Number of Layers', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('ReLU vs SNN Accuracy Across Network Depths', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(80, 100)

plt.tight_layout()
plt.savefig('depth_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'depth_comparison.png'")
plt.show()

# Print summary
print("\n" + "="*60)
print("DEPTH COMPARISON RESULTS")
print("="*60)
print(f"{'Layers':<10} {'ReLU':<12} {'SNN':<12} {'Accuracy Drop':<15}")
print("-" * 60)
for n in layers:
    print(f"{n:<10} {results[n]['relu_acc']:<12.2f} {results[n]['snn_acc']:<12.2f} {results[n]['drop']:<15.2f}")

# NEW: Add spike distribution plot
print("\n" + "="*60)
print("Creating spike distribution visualization...")
print("="*60)

try:
    with open('spike_times_data.pkl', 'rb') as f:
        spike_data = pickle.load(f)
    
    # Create spike distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    #is spike data exist create side by side camparison of 2-layer and 8-layer distributions
    if 2 in spike_data:
        ax1.hist(spike_data[2], bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_title('2-Layer: Well-Distributed Spike Times', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Spike Time', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.axvline(np.mean(spike_data[2]), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(spike_data[2]):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if 8 in spike_data:
        ax2.hist(spike_data[8], bins=30, color='red', alpha=0.7, edgecolor='black')
        ax2.set_title('8-Layer: Clustering Near t_max', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Spike Time', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.axvline(np.mean(spike_data[8]), color='darkred', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(spike_data[8]):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spike_distributions.png', dpi=150, bbox_inches='tight')
    print("Spike distribution plot saved as 'spike_distributions.png'")
    plt.show()
    
except FileNotFoundError:
    print("spike_times_data.pkl not found. Run convert_all_models.py first to generate spike data.")