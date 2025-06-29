# Neural Network from Scratch 🧠

A complete implementation of a neural network from scratch using only NumPy, inspired by the simple MNIST neural network but with many additional features and improvements.

## 🌟 Features

- **Pure NumPy Implementation**: No TensorFlow, PyTorch, or other ML frameworks
- **Flexible Architecture**: Configurable number of hidden layers and neurons
- **Advanced Training**: Mini-batch gradient descent with validation monitoring
- **Comprehensive Evaluation**: Training history, accuracy metrics, and visualizations
- **Model Persistence**: Save and load trained models
- **Robustness Testing**: Evaluate model performance with noisy inputs
- **Multiple Experiments**: Architecture comparison, learning rate optimization, and more

## 🚀 Quick Start

### Installation

```bash
# Clone or download the files
# Install required dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from mnist_neural_network import NeuralNetwork, load_and_preprocess_mnist

# Load MNIST data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_mnist()

# Create neural network
nn = NeuralNetwork(
    input_size=784,
    hidden_sizes=[128, 64],  # Two hidden layers
    output_size=10,
    learning_rate=0.01
)

# Train the network
nn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64)

# Evaluate
test_accuracy = nn.calculate_accuracy(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### Run Examples

```bash
# Quick demo (recommended first run)
python simple_example.py

# Full training with all features
python mnist_neural_network.py

# Advanced experiments and analysis
python demo_advanced_features.py
```

## 📊 What's Included

### Core Neural Network (`mnist_neural_network.py`)
- Complete feedforward neural network implementation
- ReLU activation for hidden layers, Softmax for output
- Cross-entropy loss function
- Backpropagation algorithm
- Mini-batch gradient descent
- Training history tracking
- Model saving/loading functionality

### Simple Example (`simple_example.py`)
- Quick 5-minute demo
- Basic usage patterns
- Visualization of predictions
- Model persistence example

### Advanced Features (`demo_advanced_features.py`)
- **Architecture Experiments**: Compare different network sizes
- **Learning Rate Optimization**: Find optimal learning rates
- **Training Dynamics Analysis**: Detailed convergence analysis
- **Robustness Testing**: Evaluate performance with noisy inputs

## 🏗️ Architecture

The neural network supports flexible architectures:

```
Input Layer (784) → Hidden Layer(s) → Output Layer (10)
                    [Configurable]
```

**Default Architecture:**
- Input: 784 neurons (28×28 MNIST images)
- Hidden Layer 1: 128 neurons (ReLU activation)
- Hidden Layer 2: 64 neurons (ReLU activation)  
- Output: 10 neurons (Softmax activation)
- Total Parameters: ~109,000

## 🎯 Performance

Expected performance on MNIST:
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~97%
- **Test Accuracy**: ~96-97%
- **Training Time**: ~2-3 minutes (full dataset, 100 epochs)

## 🔧 Customization

### Network Architecture
```python
# Single hidden layer
nn = NeuralNetwork(hidden_sizes=[256])

# Deep network
nn = NeuralNetwork(hidden_sizes=[512, 256, 128, 64])

# Wide network
nn = NeuralNetwork(hidden_sizes=[1024, 512])
```

### Training Parameters
```python
nn.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,           # Number of training epochs
    batch_size=32,       # Mini-batch size
    verbose=True         # Print training progress
)
```

### Learning Rate
```python
# Conservative learning
nn = NeuralNetwork(learning_rate=0.001)

# Aggressive learning
nn = NeuralNetwork(learning_rate=0.1)
```

## 📈 Visualizations

The implementation includes several visualization tools:

1. **Training History**: Loss and accuracy curves
2. **Prediction Samples**: Visual comparison of predictions vs. true labels
3. **Architecture Comparison**: Performance across different network sizes
4. **Learning Rate Analysis**: Convergence behavior with different learning rates
5. **Robustness Testing**: Performance degradation with input noise

## 🧪 Experiments

### Architecture Comparison
```python
from demo_advanced_features import experiment_with_architectures
results = experiment_with_architectures()
```

### Learning Rate Optimization
```python
from demo_advanced_features import experiment_with_learning_rates
results = experiment_with_learning_rates()
```

### Robustness Testing
```python
from demo_advanced_features import test_robustness
results = test_robustness()
```

## 💾 Model Persistence

```python
# Save trained model
nn.save_model('my_model.pkl')

# Load model
nn_loaded = NeuralNetwork()
nn_loaded.load_model('my_model.pkl')

# Use loaded model
predictions = nn_loaded.predict(X_test)
```

## 🔍 Key Improvements Over Basic Implementation

1. **Flexible Architecture**: Support for any number of hidden layers
2. **Proper Weight Initialization**: Xavier/Glorot initialization
3. **Mini-batch Training**: More stable and faster convergence
4. **Validation Monitoring**: Prevent overfitting
5. **Comprehensive Metrics**: Detailed performance analysis
6. **Visualization Tools**: Multiple plotting functions
7. **Model Persistence**: Save/load functionality
8. **Robustness Testing**: Noise resilience evaluation
9. **Experiment Framework**: Systematic hyperparameter testing
10. **Clean Code Structure**: Well-documented and modular

## 📚 Educational Value

This implementation is perfect for:
- **Learning**: Understanding neural networks from first principles
- **Teaching**: Clear, well-commented code for instruction
- **Experimentation**: Easy to modify and extend
- **Research**: Baseline implementation for custom algorithms

## 🤝 Contributing

Feel free to:
- Add new activation functions
- Implement different optimizers (Adam, RMSprop)
- Add regularization techniques (dropout, L1/L2)
- Extend to other datasets
- Improve visualizations

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by the Kaggle notebook: "Simple MNIST NN from Scratch (NumPy, no TF/Keras)"
- MNIST dataset provided by Yann LeCun et al.
- Built with NumPy, Matplotlib, and Scikit-learn

---

**Happy Learning! 🎓**

*This implementation demonstrates that you don't always need complex frameworks to build effective neural networks. Sometimes, understanding the fundamentals with pure NumPy is the best way to learn!* 