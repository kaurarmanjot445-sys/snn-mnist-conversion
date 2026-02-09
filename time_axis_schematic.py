
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 6))  #single layer for schematic diagram

# Layer 1
ax.add_patch(patches.Rectangle((0, 2), 1.0, 0.5, 
                                facecolor='lightblue', edgecolor='blue', linewidth=2))
ax.text(0.5, 2.25, 'Layer 1\nSpike Processing', ha='center', va='center', fontsize=11, fontweight='bold')
ax.arrow(1.0, 2.25, 0.3, 0, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
ax.text(1.15, 2.6, 't_max(L1) = t_min(L2)', fontsize=10, color='red', fontweight='bold')

# Layer 2
ax.add_patch(patches.Rectangle((1.5, 2), 1.0, 0.5,
                                facecolor='lightcoral', edgecolor='red', linewidth=2))
ax.text(2.0, 2.25, 'Layer 2\nSpike Processing', ha='center', va='center', fontsize=11, fontweight='bold')
ax.arrow(2.5, 2.25, 0.3, 0, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
ax.text(2.65, 2.6, 't_max(L2) = t_min(L3)', fontsize=10, color='red', fontweight='bold')

# Layer 3
ax.add_patch(patches.Rectangle((3.0, 2), 1.0, 0.5,
                                facecolor='lightgreen', edgecolor='green', linewidth=2))
ax.text(3.5, 2.25, 'Layer 3\nOutput', ha='center', va='center', fontsize=11, fontweight='bold')

# Time axis(horizontal time axis below layers)
ax.plot([0, 4], [1.5, 1.5], 'k-', linewidth=2)
ax.plot([0, 0], [1.4, 1.6], 'k-', linewidth=2)
ax.text(0, 1.2, 't=0', ha='center', fontsize=10)
#add tick marks for key time points
for t in [1.0, 1.5, 2.5, 3.0, 4.0]:
    ax.plot([t, t], [1.4, 1.6], 'k-', linewidth=1.5)
    ax.text(t, 1.2, f't={t}', ha='center', fontsize=9)

ax.text(2, 1.0, 'Time →', ha='center', fontsize=12, fontweight='bold')

# Annotations
ax.text(0.5, 3.2, 'Input spikes\nencoded', ha='center', fontsize=9, style='italic')
ax.text(2.0, 3.2, 'Hidden layer\nprocessing', ha='center', fontsize=9, style='italic')
ax.text(3.5, 3.2, 'Output\nclassification', ha='center', fontsize=9, style='italic')

#finalize and save
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(0.5, 3.5)
ax.axis('off')
ax.set_title('Multi-Layer SNN: Temporal Propagation', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('time_axis_schematic.png', dpi=150, bbox_inches='tight')
print("Saved time_axis_schematic.png")
plt.show()