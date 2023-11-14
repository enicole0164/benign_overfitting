import tensorflow as tf
import numpy as np

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
omega_init = 1.0

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

# Bring data from the npz file
n, p, mu2, eta = 10, 600, 10000.0, 0.1
data = np.load(f'./generated_samples/{n}_{p}_{mu2}_{eta}.npz')
X, y = data['arr_0'], data['arr_1'].astype(np.float32)

# Specify the learning rate
learning_rate = 0.001

# Create an instance of the SGD optimizer with the specified learning rate
custom_optimizer = tf.keras.optimizers.SGD(learning_rate)

# Instantiate the model
hidden_units = 64
model = CustomModel(hidden_units, activation_function=smoothed_leaky_relu,
                    weight_initializer_1=weight_initializer_first_layer,
                    weight_initializer_2=weight_initializer_second_layer)

# Compile the model with the custom loss
model.compile(optimizer=custom_optimizer, loss=custom_loss)

# Train the model
model.fit(X, y, epochs=10000, batch_size=32)
