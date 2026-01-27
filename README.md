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
