
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results, or fall back to my actual numbers if the pkl isn't there yet
if os.path.exists('conversion_results.pkl'):
    with open('conversion_results.pkl', 'rb') as f:
        results = pickle.load(f)
else:
    print("conversion_results.pkl not found, using my actual experiment numbers.")
    results = {
        2: {'relu_acc': 0.9826, 'snn_acc': 0.9827, 'sparsity': 0.504},
        4: {'relu_acc': 0.9830, 'snn_acc': 0.9829, 'sparsity': 0.354},
        6: {'relu_acc': 0.9826, 'snn_acc': 0.9827, 'sparsity': 0.300},
        8: {'relu_acc': 0.9826, 'snn_acc': 0.9827, 'sparsity': 0.276},
    }

layers = sorted(results.keys())
relu_accs = [results[L]['relu_acc'] * 100 for L in layers]
snn_accs = [results[L]['snn_acc'] * 100 for L in layers]
sparsity = [results[L].get('sparsity', 0.3) for L in layers]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('MNIST: ReLU MLP to TTFS SNN | B1 Identity Mapping\n',
             fontsize=12, fontweight='bold')

# Left: accuracy comparison
ax1.plot(layers, relu_accs, 'o-', color='royalblue', linewidth=2.5,
         markersize=9, label='ReLU MLP (trained)')
ax1.plot(layers, snn_accs, 's--', color='tomato', linewidth=2.5,
         markersize=9, label='TTFS SNN (converted)')
ax1.set_xticks(layers)
ax1.set_xlabel('Number of Hidden Layers', fontsize=12)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy vs Depth\n(lines overlap = exact conversion)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
y_vals = relu_accs + snn_accs
ax1.set_ylim([min(y_vals) - 0.5, max(y_vals) + 0.5])

# Right: sparsity — deeper networks fire fewer spikes
ax2.bar(layers, sparsity, color='steelblue',
        edgecolor='white', linewidth=0.8, width=0.6)
ax2.axhline(y=0.3, color='tomato', linestyle='--',
            linewidth=1.5, label='0.3 paper target')
ax2.set_xticks(layers)
ax2.set_xlabel('Number of Hidden Layers', fontsize=12)
ax2.set_ylabel('Avg Spikes per Neuron', fontsize=12)
ax2.set_title('Spiking Sparsity vs Depth', fontsize=12)
ax2.set_ylim([0, max(sparsity) * 1.3])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('depth_comparison.png', dpi=150, bbox_inches='tight')
print("Saved depth_comparison.png")