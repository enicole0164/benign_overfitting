from tempfile import TemporaryFile
import numpy as np
from sklearn.model_selection import train_test_split

def generate_samples(n, p, eta):
    # Generate p-dimensional mean vector for Pclust
    mu = np.random.normal(loc=0, scale=1, size=p)

    mu = np.sqrt(5.0 / (np.linalg.norm(mu))**2) * mu
    
    # Step 1: Sample clean labels y~ ∈ {±1} uniformly at random
    y_tilde = np.random.choice([-1, 1], size=(n, 1))
    
    # Step 2: Sample z ∼ Pclust
    z = np.random.normal(loc=0, scale=1, size=(n, p))
    
    # Step 3: Generate x_tilde = z + y_tilde * μ
    x_tilde = z + y_tilde * mu
    
    # Step 4: Introduce noise based on the given noise rate η
    noisy_labels = np.where(np.random.rand(n, 1) < eta, -y_tilde, y_tilde)
    
    # Concatenate x_tilde and noisy_labels horizontally
    # samples = np.hstack((x_tilde, noisy_labels))

    X_train, X_test, y_train, y_test = train_test_split(x_tilde, noisy_labels, test_size=0.33, random_state=42)
    
    return X_train.T, X_test.T, y_train.T, y_test.T

np.random.seed(42)

# Example usage:
n = 50  # Number of samples
p = 12000   # Number of dimensions
# eta = 0.1  # Noise rate
eta = 0

# Generate n samples of (x, y) pairs with a shared mean vector as a NumPy array
X_train, X_test, y_train, y_test = generate_samples(n, p, eta)

# Print the generated samples
print("Save generated samples...")
np.savez('random_samples_wo_noise', X_train, X_test, y_train, y_test)
print(X_train)
