import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def data_visualizer(data, labels, file_name):
    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    # Create a 2D plot with different colors for each class
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[labels == -1, 0], principal_components[labels == -1, 1], label='Class -1', c='blue')
    plt.scatter(principal_components[labels == 1, 0], principal_components[labels == 1, 1], label='Class 1', c='red')

    plt.title('2D PCA Plot with Class Colors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(file_name, format = 'png')
    plt.show()
    # plt.savefig(file_name, format = 'png')

# # Example usage
# # Replace the following with your actual data and labels
# # For example, data = np.random.rand(100, 5) and labels = np.random.choice([-1, 1], size=(100,))
# data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# labels = np.array([-1, 1, -1])  # Replace with your actual labels

# data_visualizer(data, labels)


# Example usage:
n = 10  # Number of samples
p = 600   # Number of dimensions
mu2 = 10000.0 # Squared norm of mu
eta = 0  # Noise rate


# Example usage
# Replace the following with your actual data
# For example, data = np.random.rand(100, 5) to generate random data with 100 samples and 5 features
random_samples = np.load(f'./generated_samples/{n}_{p}_{mu2}_{eta}.npz')
X_train = random_samples['arr_0'].T
X_test = random_samples['arr_1'].T
y_train = random_samples['arr_2'].T.flatten()
y_test = random_samples['arr_3'].T

print(X_train.shape)
print(y_train.flatten().shape)

file_name = f'./generated_samples/plots/PCA_{n}_{p}_{mu2}_{eta}.png'

data_visualizer(X_train, y_train, file_name)

