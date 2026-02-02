
"Implementation based on the exact gradient equivalence framework (Bellec et al.)."
**Note: MNIST accuracy limited by training duration (3 epochs).**

# snn-mnist-conversion
Time-to-first-spike SNN conversion for MNIST - exact gradient equivalence implementation

# SNN-MNIST Conversion

Implementation of time-to-first-spike spiking neural network conversion for MNIST digit classification, based on exact gradient equivalence.

# Results

**MNIST Conversion (Full Test Set):**
- ReLU Network: 85.62% accuracy
- SNN Conversion: 85.67% accuracy  
- Accuracy drop: -0.05% (effectively identical)

**Parameter Sensitivity:**
- Correct threshold scaling: maintains machine precision up to 100 layers
- 5% threshold error: breaks at ~20-30 layers

# Files

- 'train_mnist.py' - Train ReLU MLP on MNIST
- 'convert_mnist_snn.py' - Convert to SNN and test on 10k samples
- '2_layer_test.py' - Basic 2-layer conversion validation
- 'stability_analysis.py' - Parameter sensitivity analysis

## Usage

### Train model
''bash
python train_mnist_simple.py

*Test SNN conversion
python convert_snn_simple.py

#Requirements
pip install numpy tensorflow

#Method
Uses parameter mapping from exact gradient equivalence:
V = t_max - b - J.sum(0)
α = 1 - J.sum(0)
Activities encoded as spike times: high activity → early spike, low activity → late spike.

#Architecture
3-layer MLP: 784 → 400 → 400 → 10
Training: 3 epochs, batch size 64



**Extended Multi-Layer MNIST Analysis**
This section extends the original single-layer and shallow-network validation to variable-depth ReLU MLPs, following Prof. Bellec’ suggestion to study depth-dependent behavior of the conversion.

# Multi-Layer Results (MNIST)

    Layers	 Architecture	        ReLU Acc	   SNN Acc	   Drop
		 2	     784 → 400 → 10       96.90%       96.80%      0.10%
				
		 4	    784 → 400^3 → 10      96.80%       95.10%      1.70%
				
     6      784 → 400^5 → 10      96.60%       92.80%      3.80%

     8      784 → 400^7 → 10      95.80%       83.50%      12.30%

referenced as :depth_comparison.png

# Training performance: All ReLU networks reached ~98% training accuracy using Adam (15 epochs).

**Interpretation**
1. **Shallow networks* (2–4 layers)*:Conversion preserves classification performance with minimal degradation (<2%).
2. **Moderate depth* (6 layers)*:Accuracy drop becomes noticeable but remains within a reasonable range.
3. **Deep networks* (8 layers)*:Significant degradation appears, indicating numerical instability in deep spike-time propagation.
   
**These results suggest that while exact gradient equivalence holds mathematically, numerical precision effects accumulate with depth in time-to-first-spike implementations.**

# Discussion
**The observed degradation in deep networks is likely due to:*
1.Accumulation of floating-point errors across layers
2.Spike times clustering near t_max, amplifying small numerical differences
3.Increased sensitivity to weight magnitude and normalization in deep SNNs
**This behavior is consistent with known challenges in rate-to-spike and time-based conversion methods and highlights an important practical limitation worth further investigation.**

**Method Summary**
# Encoding:
High activation → early spike
Low activation → late spike

# Parameter mapping (Bellec et al.):
V = t_max − b − ΣW
α = 1 − ΣW

# Architecture:
Input: 784 (MNIST)
Hidden layers: 400 units (variable depth)
Output: 10 classes

# Training:
Optimizer: Adam (lr = 1e-3)
Epochs: 15
Batch size: 64
Framework: PyTorch

# Files (Extended)
1.train_mnist_pytorch.py – Train variable-depth ReLU MLPs
2.convert_all_models.py – Batch SNN conversion and evaluation
3.create_plot.py – Generate depth vs accuracy plot
4.depth_comparison.png – Visualization
 # Takeaway
# This extension shows that ReLU-to-SNN conversion is robust for shallow and moderately deep networks, while deep architectures expose numerical limits of spike-time propagation, making depth a key factor in practical SNN conversion.
