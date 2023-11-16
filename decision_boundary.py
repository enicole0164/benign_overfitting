import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf

# Bring data from the npz file
n, p, mu2, eta = 100, 80000, 8.0, 0

data = np.load(f'./generated_samples/{n}_{p}_{mu2}_{eta}.npz')
X, y = data['arr_0'], data['arr_1'].astype(np.float32)

# The initialization variance for the first layer
omega_init = 3.45267E-14
learning_rate = 0.001
random_seed = 42

class CustomModel(tf.keras.Model):
    def __init__(self, hidden_units, activation_function, weight_initializer_1, weight_initializer_2):
        super(CustomModel, self).__init__()

        # Define layers
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation=activation_function,
                                           kernel_initializer=weight_initializer_1)
        self.dense2 = tf.keras.layers.Dense(1, activation='linear',
                                           kernel_initializer=weight_initializer_2, trainable=False)

    def call(self, inputs):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

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

def create_custom_model():
    hidden_units = 64
    return CustomModel(hidden_units, activation_function=smoothed_leaky_relu,
                       weight_initializer_1=weight_initializer_first_layer,
                       weight_initializer_2=weight_initializer_second_layer)

# Load your saved model
model = keras.models.load_model(f'./model/{n}_{p}_{mu2}_{eta}_{omega_init}_{learning_rate}_{random_seed}',
                                custom_objects={"custom_loss": custom_loss})


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Apply PCA to reduce the dimensionality
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot the decision boundary in the reduced space
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
meshgrid_points = np.c_[xx.ravel(), yy.ravel()]

meshgrid_points_original = pca.inverse_transform(meshgrid_points)

Z = model.predict(meshgrid_points_original)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundary in 2D PCA Space')
plt.show()
