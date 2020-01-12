import numpy as np
import matplotlib.pyplot as plt


def kernel_function(x, x_p, s_f=1, l=1):
    # get || x - x_p || ^ 2
    squared_distance = np.sum(x ** 2, axis=1).reshape(-1, 1) + np.sum(x_p ** 2, axis=1) \
                       - 2 * np.matmul(x, np.transpose(x_p))
    # compute and return kernel function
    return (s_f ** 2) * np.exp((-1 / (2 * l ** 2)) * squared_distance)


def plot_gaussian_process(x, dist):
    # iterate through each distribution
    for i in range(dist.shape[1]):
        # plot each distribution
        plt.plot(x, dist[:, i])
    plt.show()


def prior_functions():
    # set up minimum and maximum intervals
    interval_min, interval_max = -4, 4
    # take arbitrary size of n = 50
    n = 50
    # generate X values
    x = np.linspace(interval_min, interval_max, n).reshape(-1, 1)
    # compute kernel function for given x
    kernel = kernel_function(x, x)

    # compute cholesky decomposition of kernel matrix to get L matrix
    l_mat = np.linalg.cholesky(kernel + 1e-6 * np.eye(n))
    # generate 10 prior distributions
    prior = np.dot(l_mat, np.random.normal(size=(n, 10)))
    return x, prior


def create_training_data():
    # assign x values
    x = np.array([-3.8, -3.2, -3, 1, 3])
    # assign y values
    # y = sin(0.5 * x)
    y = np.array([-0.9463, -0.9996, -0.9975, 0.4794, 0.9975])
    return x.reshape(-1, 1), y.reshape(-1, 1)


def plot_mean_cov_gt(mean, cov, x, y, x_star):
    # plot training samples
    plt.plot(x, y, 'rx', label='training samples')
    # plot ground truth curve
    plt.plot(x_star, np.sin(x_star / 2), label='ground truth')
    # plot estimated mean curve
    plt.plot(x_star, mean, label='predicted')
    # get standard deviation from diagonal elements of covariance matrix
    sigma = np.sqrt(np.diag(cov))
    region = 1.96 * sigma
    # fill error regions for estimated mean curve
    plt.fill_between(x_star.flatten(), mean.flatten() + region, mean.flatten() - region, alpha=0.2)
    plt.legend(loc='upper left')
    plt.show()


def non_linear_regression(train_X, train_y, x):
    # compute kernel function of train_X, train_X
    k = kernel_function(train_X, train_X)
    # compute kernel function of train_X, x
    k_star = kernel_function(train_X, x)
    # compute kernel function of x, x
    k_star_star = kernel_function(x, x)

    # compute cholesky decomposition of k to get l
    l = np.linalg.cholesky(k + 1e-6 * np.eye(len(train_X)))
    # solve for m using l and train_y
    m = np.linalg.solve(l, train_y)
    # solve for alpha using l.T and m
    alpha = np.linalg.solve(np.transpose(l), m)
    # estimate mean by multiplying k_star and alpha
    mean = np.matmul(np.transpose(k_star), alpha)

    # solve for v using l and k_star
    v = np.linalg.solve(l, k_star)
    # estimate covariance matrix from k_star_star and v
    cov = k_star_star - np.matmul(np.transpose(v), v)
    return mean, cov


def compute_posterior(mean, cov, n):
    # compute cholesky decomposition of covariance matrix to get L matrix
    l_mat = np.linalg.cholesky(cov + 1e-6 * np.eye(n))
    # generate 10 posterior distributions with estimated mean and covariance
    posterior = mean + np.dot(l_mat, np.random.normal(size=(n, 10)))
    return posterior


def main():
    # set up random seed for reproducibility
    np.random.seed(123)

    # a) prior functions
    # generate prior functions
    x, prior = prior_functions()
    # plot the gaussian process prior
    plot_gaussian_process(x, prior)

    # b) estimate mean and error
    # create training data set
    train_X, train_y = create_training_data()
    # compute mean and covariance
    mean, cov = non_linear_regression(train_X, train_y, x)
    # plot predicted mean, error curves and ground truth
    plot_mean_cov_gt(mean, cov, train_X, train_y, x)

    # c) posterior functions
    posterior = compute_posterior(mean, cov, x.shape[0])
    # plot gaussian process posterior
    plot_gaussian_process(x, posterior)


if __name__ == '__main__':
    main()
