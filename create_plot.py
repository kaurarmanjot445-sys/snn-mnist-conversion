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
