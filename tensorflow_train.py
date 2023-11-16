import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import os.path

# Set a random seed
random_seed = 42
np.random.seed(random_seed)

class CustomModel(tf.keras.Model):
    def __init__(self, hidden_units, activation_function, weight_initializer_1, weight_initializer_2):
        super(CustomModel, self).__init__()
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.weight_initializer_1 = weight_initializer_1
        self.weight_initializer_2 = weight_initializer_2

        # Define layers
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation=activation_function,
                                           kernel_initializer=weight_initializer_1)
        self.dense2 = tf.keras.layers.Dense(1, activation='linear',
                                           kernel_initializer=weight_initializer_2, trainable=False)

    def call(self, inputs):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output
    
    # def get_config(self):
    #     base_config = super().get_config()
    #     config = {
    #         "hidden_units" : self.hidden_units,
    #         "activation_function" : self.activation_function,
    #         "weight_initializer_1" : self.weight_initializer_1,
    #         "weight_initializer_2" : self.weight_initializer_2
    #     }
    #     return config

# Custom Smoothed Leaky ReLU activation function
def smoothed_leaky_relu(x, gamma=0.5, H=2):
    condition1 = tf.math.greater_equal(x, 1/H)
    condition2 = tf.math.less(x, -1/H)

    return tf.where(condition1, 
                    x - (1 - gamma) / (4 * H), 
                    tf.where(condition2,
                             gamma * x - (1 - gamma) / (4 * H),
                             (1 - gamma) * H * x**2 / 4 + (1 + gamma) * x / 2))

# Custom loss function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(1 + tf.math.exp(-y_true*y_pred)))

# The initialization variance for the first layer
omega_init = 3.45267E-14

# Custom weight initializer for the first layer
def weight_initializer_first_layer(shape, dtype=None):
    return tf.constant(np.random.normal(0, omega_init, size=shape).astype(np.float32))

# Custom weight initializer for the second layer
def weight_initializer_second_layer(shape, dtype=None):
    return tf.constant(np.random.uniform(-1/np.sqrt(shape[0]), 1/np.sqrt(shape[0]), size=shape).astype(np.float32))

# # Generate some dummy data
# data_size = 100
# input_data = np.random.randn(data_size, 10).astype(np.float32)
# labels = np.random.choice([-1, 1], size=(data_size, 1)).astype(np.float32)

mu2 = [128.0, 256.0]
eta = [0.1]

# Create a meshgrid from mu2 and eta
mu2_mesh, eta_mesh = np.meshgrid(mu2, eta, indexing='ij')

# Flatten the meshgrid arrays to get all combinations
mu2_eta_combinations = np.column_stack((mu2_mesh.flatten(), eta_mesh.flatten()))

# Iterate over all combinations
for mu2_val, eta_val in mu2_eta_combinations:
    if eta_val == 0.0: eta_val = 0
    np.random.seed(random_seed)

    # Bring data from the npz file
    n, p, mu2, eta = 100, 80000, mu2_val, eta_val

    # Specify the learning rate and epoch
    learning_rate = 0.001
    epoch = 10000

    # model_path = f"./model/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}"
    # if os.path.exists(model_path):
    #     print(f"Model Exists {n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}")
    #     continue

    print(f'mu2: {mu2_val}, eta: {eta_val}')
    # Your code for each combination goes here
    # For example, you can call a function or perform some computations
    
    data = np.load(f'./generated_samples/{n}_{p}_{mu2}_{eta}.npz')
    X, y = data['arr_0'], data['arr_1'].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Create an instance of the SGD optimizer with the specified learning rate
    custom_optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Instantiate the model
    hidden_units = 64
    model = CustomModel(hidden_units, activation_function=smoothed_leaky_relu,
                        weight_initializer_1=weight_initializer_first_layer,
                        weight_initializer_2=weight_initializer_second_layer,
                        )

    # Compile the model with the custom loss
    model.compile(optimizer=custom_optimizer, loss=custom_loss)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=y_train.shape[0])

    # Save the model to a file
    model.save(f'./model/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}')

    # Evaluate the model on the test set
    y_pred = np.sign(model.predict(X_test))

    train_accuracy = np.mean(np.equal(y_train, np.sign(model.predict(X_train))))
    test_accuracy = np.mean(np.equal(y_test, y_pred))

    # Retrive y_loss 
    y_loss = history.history['loss']
    np.save(f'./loss_history/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}.npy', y_loss)

    x_len = np.arange(len(y_loss))

    # Clear the current figure
    plt.clf()  # or plt.cla()

    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'./loss_plot/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}.png')

    # Print the test accuracy
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')

    # Print the first loss
    print(f'First Loss: {y_loss[0]}')

    np.save(f'./accuracy/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}.npy', np.array([train_accuracy, test_accuracy]))
    np.savetxt(f'./accuracy/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}.txt', np.array([train_accuracy, test_accuracy]))
    np.savetxt(f'./first_last_loss/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}.txt', np.array([y_loss[0], y_loss[-1]]))

    ############-------------PCA-------------##############
    # Clear the current figure
    plt.clf()  # or plt.cla()

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_scaled = scaler.transform(X_train)  
    X_test_scaled = scaler.transform(X_test)  

    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=12)
    X_pca = pca.fit_transform(X_scaled)
    X_train_pca = pca.transform(X_train_scaled)  
    X_test_pca = pca.transform(X_test_scaled)  

    # Plot the decision boundary in the reduced space
    x_min, x_max = X_pca[:, 10].min() - 5, X_pca[:, 10].max() + 5
    y_min, y_max = X_pca[:, 11].min() - 5, X_pca[:, 11].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 5), np.arange(y_min, y_max, 5))
    meshgrid_points = np.c_[xx.ravel(), yy.ravel()]

    # meshgrid_points_original = pca.inverse_transform(meshgrid_points)
    meshgrid_points_original = pca.inverse_transform(np.concatenate((np.zeros((1190, 10)), meshgrid_points), axis=1))
    meshgrid_points_original = scaler.inverse_transform(meshgrid_points_original)

    Z = np.sign(model.predict(meshgrid_points_original))
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X_train_pca[:, 10], X_train_pca[:, 11], c=y_train, cmap=plt.cm.RdYlBu)
    plt.scatter(X_test_pca[:, 10], X_test_pca[:, 11], c=y_test, cmap=plt.cm.RdYlBu)
    plt.xlabel('Principal Component 10')
    plt.ylabel('Principal Component 11')
    plt.title('Decision Boundary in 2D PCA Space')
    plt.savefig(f'./decision_boundary/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}_10_11_pc.png')


    # tf.keras.backend.clear_session()