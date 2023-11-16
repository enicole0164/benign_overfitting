import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def data_visualizer(data, labels, file_name):
    np.random.seed(42)
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

# Example usage:
n = 100  # Number of samples
p = 102400   # Number of dimensions
mu2 = 512.0 # Squared norm of mu
eta = 0.1  # Noise rate


# Example usage
# Replace the following with your actual data
# For example, data = np.random.rand(100, 5) to generate random data with 100 samples and 5 features
random_samples = np.load(f'./generated_samples/{n}_{p}_{mu2}_{eta}.npz')
X = random_samples['arr_0']
y = random_samples['arr_1'].T.flatten()

file_name = f'./generated_samples/plots/PCA_{n}_{p}_{mu2}_{eta}.png'

data_visualizer(X, y, file_name)

