import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def data_visualizer(data, labels, file_name):
    # Fit Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(n_components=1)
    print(data.shape)
    print(labels.shape)
    X_lda = lda.fit_transform(data, labels)

    print(X_lda.shape)

    fl_labels = y.T.flatten()

    # Create a 2D plot with different colors for each class
    plt.figure(figsize=(8, 6))
    if X_lda.shape[1] == 1:
        plt.scatter(X_lda[fl_labels == -1], np.zeros_like(X_lda[fl_labels == -1]), label='Class -1', c='blue')
        plt.scatter(X_lda[fl_labels == 1], np.zeros_like(X_lda[fl_labels == 1]), label='Class 1', c='red')
    else:
        plt.scatter(X_lda[fl_labels == -1, 0], X_lda[fl_labels == -1, 1], label='Class -1', c='blue')
        plt.scatter(X_lda[fl_labels == 1, 0], X_lda[fl_labels == 1, 1], label='Class 1', c='red')

    plt.title('LDA Visualization')
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.legend()
    plt.savefig(file_name, format = 'png')
    plt.show()

# Parameters
n = 100  # Number of samples
p = 80000   # Number of dimensions
mu2 = 8.0 # Squared norm of mu
eta = 0.01  # Noise rate

# Load data
random_samples = np.load(f'./generated_samples/{n}_{p}_{mu2}_{eta}.npz')
X = random_samples['arr_0']
y = random_samples['arr_1'].flatten()

file_name = f'./generated_samples/plots/LDA_{n}_{p}_{mu2}_{eta}.png'

data_visualizer(X, y, file_name)
