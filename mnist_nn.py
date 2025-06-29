import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import time

class NeuralNetwork:
    """
    A neural network implementation from scratch using only NumPy
    Based on the simple MNIST NN but with additional features
    """
    
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.01):
        """
        Initialize the neural network
        
        Args:
            input_size: Number of input features (784 for MNIST 28x28 images)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes (10 for MNIST digits)
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Create layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # For tracking training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        """Cross-entropy loss function"""
        m = y_true.shape[0]
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / m
    
    def one_hot_encode(self, y, num_classes):
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        
        Args:
            X: Input data
            
        Returns:
            activations: List of activations for each layer
            z_values: List of pre-activation values
        """
        activations = [X]
        z_values = []
        
        current_input = X
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.relu(z)
            activations.append(a)
            current_input = a
        
        # Output layer (softmax)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        z_values.append(z_output)
        output = self.softmax(z_output)
        activations.append(output)
        
        return activations, z_values
    
    def backward_propagation(self, X, y, activations, z_values):
        """
        Backward propagation to compute gradients
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            activations: Activations from forward pass
            z_values: Pre-activation values from forward pass
            
        Returns:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        m = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        
        # Output layer error
        delta = activations[-1] - y
        
        # Gradients for output layer
        weight_grad = np.dot(activations[-2].T, delta) / m
        bias_grad = np.sum(delta, axis=0, keepdims=True) / m
        
        weight_gradients.insert(0, weight_grad)
        bias_gradients.insert(0, bias_grad)
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self.relu_derivative(z_values[i])
            
            weight_grad = np.dot(activations[i].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def predict(self, X):
        """Make predictions on input data"""
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]
    
    def calculate_accuracy(self, X, y):
        """Calculate accuracy on given data"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress
        """
        # Convert labels to one-hot encoding
        y_train_onehot = self.one_hot_encode(y_train, self.output_size)
        if y_val is not None:
            y_val_onehot = self.one_hot_encode(y_val, self.output_size)
        
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward propagation
                activations, z_values = self.forward_propagation(X_batch)
                
                # Calculate loss
                batch_loss = self.cross_entropy_loss(y_batch, activations[-1])
                epoch_loss += batch_loss
                
                # Backward propagation
                weight_grads, bias_grads = self.backward_propagation(X_batch, y_batch, activations, z_values)
                
                # Update parameters
                self.update_parameters(weight_grads, bias_grads)
            
            # Calculate metrics
            avg_loss = epoch_loss / n_batches
            train_acc = self.calculate_accuracy(X_train, y_train)
            
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward_propagation(X_val)
                val_loss = self.cross_entropy_loss(y_val_onehot, val_activations[-1])
                val_acc = self.calculate_accuracy(X_val, y_val)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        if self.val_accuracies:
            ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename):
        """Save the trained model"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.input_size = model_data['input_size']
        self.hidden_sizes = model_data['hidden_sizes']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        print(f"Model loaded from {filename}")


def load_and_preprocess_mnist():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    
    # Load MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Further split training set to get validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def visualize_predictions(model, X_test, y_test, num_samples=10):
    """Visualize model predictions"""
    # Get random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Make predictions
    predictions = model.predict(X_test[indices])
    probabilities = model.predict_proba(X_test[indices])
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Reshape image
        image = X_test[indices[i]].reshape(28, 28)
        
        # Plot image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {y_test[indices[i]]}, Pred: {predictions[i]}\nConf: {np.max(probabilities[i]):.2f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the neural network"""
    print("🧠 Neural Network from Scratch - MNIST Classification")
    print("=" * 50)
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_mnist()
    
    # Create neural network
    print("\nCreating neural network...")
    nn = NeuralNetwork(
        input_size=784,
        hidden_sizes=[128, 64],  # Two hidden layers
        output_size=10,
        learning_rate=0.01
    )
    
    print(f"Network architecture: 784 -> 128 -> 64 -> 10")
    print(f"Total parameters: {sum(w.size for w in nn.weights) + sum(b.size for b in nn.biases)}")
    
    # Train the network
    print("\nStarting training...")
    start_time = time.time()
    
    nn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=64,
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = nn.calculate_accuracy(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    print("\nPlotting training history...")
    nn.plot_training_history()
    
    # Visualize some predictions
    print("\nVisualizing predictions...")
    visualize_predictions(nn, X_test, y_test)
    
    # Save the model
    print("\nSaving model...")
    nn.save_model('mnist_nn_model.pkl')
    
    print("\n🎉 Training complete!")
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main() 