import numpy as np

# Define the smoothed leaky ReLU activation function
def smooth_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Define the derivative of the smoothed leaky ReLU activation function
def smooth_leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha * np.exp(x))

# Initialize the parameters of the neural network with specific weight initialization for the second layer
def initialize_parameters(input_size, hidden_size, output_size, w_init):
    W1 = np.random.randn(hidden_size, input_size) * w_init
    W2 = np.random.uniform(-1/np.sqrt(hidden_size), 1/np.sqrt(hidden_size), (output_size, hidden_size))
    return W1, W2

# Forward propagation with smoothed leaky ReLU activation
def forward_propagation(X, W1, W2):
    Z1 = np.dot(W1, X)
    A1 = smooth_leaky_relu(Z1)
    Z2 = np.dot(W2, A1)
    A2 = np.sign(Z2)
    return Z1, Z2, A2

# Compute the logistic loss for binary classification
def compute_loss(Y, Z2):
    m = Y.shape[1]  # number of examples
    loss = np.sum(np.log((1 + np.exp(- Y * Z2)))) / m
    return loss

# Backward propagation and parameter update using gradient descent
def backward_propagation(X, Y, Z1, Z2, W1, W2, learning_rate):
    m = Y.shape[1]
    dZ2 = np.sum((-Y * np.exp(-Y * Z2)) / (1 + np.exp(-Y * Z2))) / m
    # dW2 = np.dot(dZ2, A1.T) / m
    dZ1 = np.dot(W2.T, dZ2) * smooth_leaky_relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T)
    
    # Update parameters (except for the fixed second layer weightbs)
    W1 -= learning_rate * dW1
    
    return W1

# Training the neural network
def train_neural_network(X, Y, hidden_size, num_epochs, learning_rate):
    input_size = X.shape[0]
    output_size = Y.shape[0]
    
    # Initialize parameters
    W1, W2 = initialize_parameters(input_size, hidden_size, output_size, 1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward propagation
        Z1, Z2, _ = forward_propagation(X, W1, W2)
        
        # Compute loss
        loss = compute_loss(Y, Z2)
        
        # Backward propagation and parameter update
        W1 = backward_propagation(X, Y, Z1, Z2, W1, W2, learning_rate)
        
        # Print the loss every 1000 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    return W1, W2

# # Sigmoid activation function
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# Example usage
# Generate random data for training
np.random.seed(42)

random_samples = np.load('random_samples.npz')
print(random_samples.files)
X_train = random_samples['arr_0']
X_test = random_samples['arr_1']
y_train = random_samples['arr_2']
y_test = random_samples['arr_3']

print(np.shape(X_train))
print(np.shape(y_train))

# Training the neural network
hidden_size = 1000
num_epochs = 100
learning_rate = 0.0001
W1, W2 = train_neural_network(X_train, y_train, hidden_size, num_epochs, learning_rate)

