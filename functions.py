import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def generate_data(num_samples=1000, radius=1.1, noise=0.3, random_seed=42):
    """
    Generates a synthetic dataset with a binary target based on a circular decision boundary.

    Args:
        num_samples (int): Number of samples to generate.
        radius (float): Radius of the circle that defines the decision boundary.
        noise (float): Standard deviation of Gaussian noise added to the data.
        random_seed (int): Seed for the random number generator for reproducibility.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Feature matrix of shape (num_samples x 2).
            - y (np.ndarray): Binary target vector of length num_samples.
    """
    np.random.seed(random_seed)

    # generate random points
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    r = radius * np.sqrt(np.random.uniform(0, 1, num_samples))

    # convert polar coordinates to Cartesian coordinates
    x1 = r * np.cos(theta) + np.random.normal(0, noise, num_samples)
    x2 = r * np.sin(theta) + np.random.normal(0, noise, num_samples)

    # calculate labels
    y = (x1**2 + x2**2 <= radius**2).astype(int)

    # stack features into an array
    X = np.column_stack((x1, x2))

    return np.hstack([X, y.reshape(-1, 1)])


def plot_boundaries(X, y, degree=1, modeltype="lr", neighbors=1):
    """
    Plots a decision boundary plot for a polynomial logistic regression

    Args:
        X (pandas dataframe): features dataframe
        y (pandas series): target series
        degree (int): degree of polynomial
    """

    if modeltype == "lr":
        model = make_pipeline(PolynomialFeatures(degree=degree), LogisticRegression())
    if modeltype == "knn":
        model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X, y)

    # get the minimum and maximum values of each feature
    x_min, x_max = X.iloc[:, 0].min() - 0.1, X.iloc[:, 0].max() + 0.1
    y_min, y_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1

    # make a mesh grid for the decision boundary visualization
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # predict probabilities on the mesh grid
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, Z, 25, cmap="RdBu", alpha=0.8)
    plt.colorbar(contour)
    plt.scatter(
        X.iloc[:, 0], X.iloc[:, 1], c=y, cmap="viridis", edgecolors="w", linewidth=1
    )
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title("Scatter Plot with Decision Boundary")
    plt.show()
