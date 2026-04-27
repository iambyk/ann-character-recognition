# Artificial Neural Networks: Character Recognition with Multi-Layer Perceptron and Logic Gate Classification with Perceptron

## ⚠️ Language Note
**This project was developed for a university-level Artificial Neural Networks course (in Turkish). The code comments and documentation are primarily in Turkish to comply with academic course requirements. However, the core algorithms, mathematical implementations, and structure are universally applicable and well-documented.**

---

## Project Overview

This project implements fundamental artificial neural network algorithms from scratch without using pre-built machine learning libraries like scikit-learn, TensorFlow, or PyTorch. The implementation focuses on understanding the core principles of neural networks through direct mathematical computation using NumPy.

### Key Features
- **Single-Layer Perceptron**: Implements the Delta Rule and Gradient Descent for binary classification
- **Multi-Layer Perceptron (MLP)**: Implements the Backpropagation algorithm for non-linear pattern recognition
- **Graphical User Interface**: Built with CustomTkinter for interactive model training and testing
- **Visualization**: Matplotlib-based decision boundary and training history plots
- **Model Persistence**: Save and load trained models in NumPy format (.npz)

---

## Project Structure

```text
artificial-neural-networks-character-recognition/
├── main.py # Application entry point
├── README.md # This file
├── requirements.txt # Python dependencies
├── src/ # Core implementation
│   ├── utils.py # Activation functions & loss functions
│   ├── perceptron.py # Single-layer Perceptron implementation
│   ├── mlp.py # Multi-layer Perceptron + Backpropagation
│   ├── data_loader.py # Dataset creation & loading
│   └── visualizer.py # Matplotlib visualizations
├── gui/ # Graphical User Interface
│   ├── __init__.py
│   └── main_gui.py # CustomTkinter main window
└── tests/ # Testing modules
    ├── test_networks.py # Network functionality tests
    └── test_visualization.py # Visualization tests
```

---

## Datasets

### Character Recognition Dataset
- **Characters**: A, B, C, D, E (5 uppercase letters)
- **Font Styles**: Regular, Italic, Bold (3 variations per character)
- **Matrix Size**: 7×5 pixels (35 input features)
- **Total Samples**: 15 examples (5 characters × 3 fonts)
- **Representation**: Binary matrices (0 and 1) flattened into vectors

### Logic Gate Dataset
- **Gates**: AND, OR, XOR
- **Input**: 2-dimensional binary vectors [x₁, x₂]
- **Output**: Binary class label (0 or 1)
- **Samples per Gate**: 4 examples (truth table)

---

## Implementation Details

### Algorithms

#### 1. Single-Layer Perceptron
Weight Update Rule:
w_i = w_i + η × (target - output) × x_i

Where:
η: Learning rate
target: Expected output
output: Actual output
x_i: Input feature

**Used for**: Linearly separable problems (AND, OR gates)

#### 2. Multi-Layer Perceptron with Backpropagation

**Forward Pass**:
z_l = W_l · a_{l-1} + b_l
a_l = σ(z_l) [Activation function]


**Backward Pass (Backpropagation)**:
δ_L = (a_L - y) ⊙ σ'(z_L)
δ_l = (W_{l+1}^T · δ_{l+1}) ⊙ σ'(z_l)

Weight Updates:
ΔW_l = -η · δ_l · a_{l-1}^T
Δb_l = -η · δ_l


**Used for**: Non-linear problems and pattern recognition (character classification)

### Activation Functions Implemented
- **Sigmoid**: σ(x) = 1 / (1 + e^(-x))
- **Tanh**: σ(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- **ReLU**: σ(x) = max(0, x)
- **Step**: σ(x) = 1 if x ≥ 0 else 0

### Loss Functions
- **Mean Squared Error (MSE)**: L = (1/n) × Σ(y - ŷ)²
- **Binary Crossentropy**: L = -(1/n) × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]

---

## Results

### Logic Gates Classification (Single-Layer Perceptron)
| Gate | Accuracy | Status |
|------|----------|--------|
| AND  | 100%     | ✓ Linearly separable - Perfect classification |
| OR   | 100%     | ✓ Linearly separable - Perfect classification |
| XOR  | ~50%     | ✗ Non-linearly separable - Requires hidden layers |

**Key Insight**: XOR cannot be solved by a single-layer Perceptron because it is not linearly separable. This demonstrates the fundamental limitation of linear classifiers and the necessity of hidden layers for non-linear problems.

### Character Recognition (MLP)
| Metric | Value |
|--------|-------|
| Training Accuracy | 100% |
| Training Samples | 15 (5 characters × 3 fonts) |
| Network Architecture | [35, 25, 5] (Input, Hidden, Output) |
| Learning Rate | 0.3 |
| Epochs | 2000 |
| Activation Function | Sigmoid |

**Observations**:
- Loss decreases exponentially during training
- Accuracy reaches 100% around epoch 400
- Model generalizes well to hand-drawn character inputs via GUI

---

## Installation & Usage

### Requirements
```bash
pip install numpy matplotlib customtkinter pillow
```
### Running the Application
```bash
python main.py
```

---

## GUI Tabs
### Character Recognition Tab

1- Adjust hyperparameters:
    Learning Rate (default: 0.3)
    Number of Epochs (default: 2000)
    Hidden Layer Neurons (default: 25)

2- Train the model: Click "Train MLP Model"

3- Draw and predict:
    Draw a character on the 7×5 grid
    Click "Predict" to get the model's classification
    View confidence scores for all classes

4- Manage models:
    Save trained weights: "Save Model"
    Load saved weights: "Load Model"

### Logic Gates Tab

Select a gate: AND, OR, or XOR
Adjust Perceptron hyperparameters
Train: Click "Train Perceptron"
View results in the output table
Visualize decision boundary: Click "Show Decision Boundary"

Key Technical Decisions
1. No Pre-built Libraries

Using only NumPy for matrix operations ensures a deep understanding of:
Matrix multiplication and broadcasting
Gradient computation and chain rule
Numerical stability considerations
Memory-efficient implementations

2. GUI Implementation

CustomTkinter provides:
Modern dark-themed interface
Cross-platform compatibility
Lightweight and responsive UI
Easy hyperparameter adjustment

3. Xavier Initialization
Hidden layer weights are initialized using:

scale = sqrt(2.0 / (n_in + n_out))
w = randn(n_in, n_out) * scale

This helps prevent vanishing/exploding gradients.

4. Numerical Stability
Sigmoid computation is clipped to prevent overflow:

x = clip(x, -500, 500)
output = 1 / (1 + exp(-x))

## Demonstration Video

A 6-8 minute demonstration video showing the complete project in action is available at:
https://youtu.be/Rczj56KlVPs

The video includes:

Project introduction and architecture overview
Live MLP training on character dataset
Character drawing and prediction demonstration
Logic gate classification (AND, OR, XOR)
Decision boundary visualization
Discussion of XOR limitations with single-layer Perceptron

## Educational Value

This project demonstrates:

Understanding Core Concepts: Direct implementation of backpropagation without abstraction
Mathematical Foundation: Practical application of calculus and linear algebra
Debugging Skills: Implementing gradient checking and loss monitoring
Software Engineering: Modular design, encapsulation, and user-friendly interface
Problem Solving: From theory to working implementation

## Limitations & Future Improvements

### Current Limitations

Small dataset (15 character examples) due to academic context
Limited to binary or simple multi-class problems
No data augmentation or normalization strategies
No mini-batch processing for efficiency

## Potential Enhancements

Add convolutional layers for better image features
Implement mini-batch gradient descent
Add regularization (L1/L2) to prevent overfitting
Extend character set to all 26 letters + digits
Implement other architectures (RNN, attention mechanisms)
Add model validation with separate test set

## Acknowledgments

Built as a learning project for university-level study of neural network fundamentals
All algorithms implemented from mathematical principles without relying on high-level ML frameworks
Developed to provide transparent, educational implementation of core ANN concepts


