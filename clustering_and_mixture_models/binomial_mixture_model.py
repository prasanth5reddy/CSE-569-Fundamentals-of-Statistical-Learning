import numpy as np
import matplotlib.pyplot as plt
import math
from clustering_and_mixture_models.k_means import get_binomial_20_flips, k_means


def comb(n, k):
    # compute n choose k value
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def e_step(x, pi, theta):
    # initialise responsibilities array with zeros of shape (1000, 3)
    resp = np.zeros((x.shape[0], pi.shape[0]))
    for i, k in enumerate(x):
        # compute responsibility for datapoint i
        p_k = pi * comb(20, k) * np.power(theta, k) * np.power(1 - theta, 20 - k)
        resp[i] = p_k / np.sum(p_k)
    # return responsibilities for all datapoints
    return resp


def m_step(resp, x):
    # compute parameter values
    pi = np.sum(resp, axis=0) / x.shape[0]
    # compute prior probabilities
    theta = np.sum(resp * x, axis=0) / (20 * np.sum(resp, axis=0))
    return pi, theta


def plot_neg_log_like(log_like_list, epochs):
    y = np.array(log_like_list)
    x = np.arange(epochs)
    plt.plot(x, y)
    plt.xlabel('No. of Iterations')
    plt.ylabel('Negative Log Likelihood')
    plt.show()


def em_binomial(points, pi, theta, epochs):
    neg_log_like_list = []
    for _ in range(epochs):
        # evaluate responsibilities
        resp = e_step(points, pi, theta)

        # estimate parameters
        pi, theta = m_step(resp, points)

        # evaluate negative log likelihood
        log_like = 0
        for _, k in enumerate(points):
            log_like += np.log(np.sum(pi * comb(20, k) * np.power(theta, k) * np.power(1 - theta, 20 - k)))
        neg_log_like_list.append(-log_like)

    print(neg_log_like_list[-1])
    plot_neg_log_like(neg_log_like_list, epochs)

    print('Prior Probabilities')
    print('pi_1 : ', pi[0])
    print('pi_2 : ', pi[1])
    print('pi_3 : ', pi[2])

    print('Parameters (Probability of Heads)')
    print('theta_1 : ', theta[0])
    print('theta_2 : ', theta[1])
    print('theta_3 : ', theta[2])


def main():
    # get points from txt file
    points = get_binomial_20_flips()
    # set k value
    k = 3
    print('K Means')
    pi_k_means, theta_k_means = k_means(points, k)

    # em_algorithm with random initialization
    np.random.seed(1)
    print('\nEM binomial with random initialization')
    # initialise mixing coefficients pi and parameters theta
    pi, theta = np.random.random(k), np.random.random(k)
    em_binomial(points, pi, theta, epochs=200)

    # em_algorithm with K means initialization
    print('\nEM binomial with K means initialization')
    em_binomial(points, pi_k_means, theta_k_means, epochs=200)


if __name__ == '__main__':
    main()
