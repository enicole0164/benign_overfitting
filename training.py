import numpy as np
import matplotlib.pyplot as plt

# # Define the smoothed leaky ReLU activation function
# def smooth_leaky_relu(x, alpha=0.01):
#     return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# # Define the derivative of the smoothed leaky ReLU activation function
# def smooth_leaky_relu_derivative(x, alpha=0.01):
#     return np.where(x > 0, 1, alpha * np.exp(x))

def smooth_leaky_relu(x, gamma = 0.5, H = 2):
    if x >= 1/H:
        return x - (1-gamma)/(4*H)
    elif x < -1/H:
        return gamma*x - (1-gamma)/(4*H)
    else:
        return (1-gamma)*H*x**2/4 + (1+gamma)*x/2

def smooth_leaky_relu_derivative(x, gamma = 0.5, H = 2):
    if x >= 1/H:
        return 1
    elif x < -1/H:
        return gamma
    else:
        return (1-gamma)*H/2 * x + (1+gamma)/2

# Initialize the parameters of the neural network with specific weight initialization for the second layer
def initialize_parameters(input_size, hidden_size, output_size, w_init):
    W1 = np.random.randn(hidden_size, input_size) * w_init
    W2 = np.random.uniform(-1/np.sqrt(hidden_size), 1/np.sqrt(hidden_size), (output_size, hidden_size))
    return W1, W2

# Forward propagation with smoothed leaky ReLU activation
def forward_propagation(X, W1, W2):
    Z1 = np.dot(W1, X)
    sl_ReLU = np.vectorize(smooth_leaky_relu)
    A1 = sl_ReLU(Z1)
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
    sl_ReLU_deriv = np.vectorize(smooth_leaky_relu_derivative)
    dZ1 = np.dot(W2.T, dZ2) * sl_ReLU_deriv(Z1)
    dW1 = np.dot(dZ1, X.T)
    
    # Update parameters (except for the fixed second layer weightbs)
    W1 -= learning_rate * dW1
    
    return W1

# Training the neural network
def train_neural_network(X, Y, hidden_size, num_epochs, learning_rate):
    input_size = X.shape[0]
    output_size = Y.shape[0]

     # List to store the loss values for each epoch
    loss_history = []
    
    # Initialize parameters
    W1, W2 = initialize_parameters(input_size, hidden_size, output_size, 1.002344217E-11)
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward propagation
        Z1, Z2, _ = forward_propagation(X, W1, W2)
        
        
        # Backward propagation and parameter update
        W1 = backward_propagation(X, Y, Z1, Z2, W1, W2, learning_rate)
        
        # Print the loss every 1000 epochs
        if epoch % 1000 == 0:
            # Compute loss
            loss = compute_loss(Y, Z2)
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
            loss_history.append(loss)
    
    # Plot the loss over epochs
    plt.plot(range(0,len(loss_history)*1000,100), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

    return W1, W2

def predict(X, W1, W2):
    _, _, A2 = forward_propagation(X, W1, W2)
    return A2

# Example usage
# Generate random data for training
# np.random.seed(42)

n = 10  # Number of samples
p = 600   # Number of dimensions
mu2 = 10.0 # Squared norm of mu
eta = 0.2  # Noise rate

random_samples = np.load( f'./generated_samples/{n}_{p}_{mu2}_{eta}.npz')
X_train = random_samples['arr_0']
X_test = random_samples['arr_1']
y_train = random_samples['arr_2']
y_test = random_samples['arr_3']


# Training the neural network
hidden_size = 1000

num_epochs = 100000
learning_rate = 0.001
W1, W2 = train_neural_network(X_train, y_train, hidden_size, num_epochs, learning_rate)

print(predict(X_test, W1, W2))
print(y_test)

np.savez(f'./models/{n}_{p}_{mu2}_{eta}', W1, W2)