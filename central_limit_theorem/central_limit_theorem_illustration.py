import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def demo_central_limit_theorem(N, n, low, high):
    # takes N = number of times the sample is drawn and n = sample size
    # return the list of mean of samples that are repeated N times
    # Let u_s be the mean of all the samples drawn for a given sample size n
    u_s = []
    for _ in range(N):
        # get random values for a sample size n
        x = np.random.randint(low, high + 1, size=n)
        # get the mean of n samples and store it
        u_s.append(np.mean(x))
    return u_s


def expected_value(x, p_x):
    # takes X = random variable and p_x = P(X = x)
    # returns expected value
    return np.sum(x * p_x)


def variance(x, p_x, u):
    # takes X = random variable and p_x = P(X = x) and u = mean
    # returns variance
    return np.sum(((x - u) ** 2) * p_x)


def show_experiment_results(N, n, var_pop, low, high):
    u_s = demo_central_limit_theorem(N=1000, n=n, low=low, high=high)
    # get the mean and variance of all the means obtained by drawing each sample size for N times
    u_x, v_x = np.mean(u_s), np.var(u_s)
    print(f'Mean for sample size n = {n} is {round(int(u_x), 4)}')
    print(f'Variance for sample size n = {n} is {round(int(v_x), 4)}')
    print(f'Population variance / n for size n = {n} is {round(var_pop / n, 4)}\n')
    return u_s


def show_plots(u_s, n, exp_val_pop, var_pop):
    u_s = sorted(u_s)
    normal = stats.norm.pdf(u_s, exp_val_pop, np.sqrt(var_pop / n))
    plt.hist(u_s, density=True)
    plt.plot(u_s, normal, '-')
    plt.title('Histogram and normal distribution for sample size = ' + str(n))
    plt.xlabel('Mean (u_s)')
    plt.show()


def main():
    # Set up dice low and high values for random distribution
    dice_dist_low, dice_dist_high = 1, 6

    # set up random values
    x = np.array([1, 2, 3, 4, 5, 6])
    # set up probability of each random value. In this case p(X = x) = 1 / 6 for all x
    p_x = np.ones_like(x) / 6

    # get expected value and variance of population
    exp_val_pop = expected_value(x, p_x)
    var_pop = variance(x, p_x, exp_val_pop)

    # print expected value and variance of population
    print('Expected value of the population : ', exp_val_pop)
    print('variance of the population : ', var_pop)

    u_s_2 = show_experiment_results(1000, 2, var_pop, dice_dist_low, dice_dist_high)
    u_s_5 = show_experiment_results(1000, 5, var_pop, dice_dist_low, dice_dist_high)
    u_s_10 = show_experiment_results(1000, 10, var_pop, dice_dist_low, dice_dist_high)
    # As we can see as n increases, mean of sample is same as mean of population
    # and variance of population / n is same as variance of sample size

    show_plots(u_s_2, 2, exp_val_pop, var_pop)
    show_plots(u_s_5, 5, exp_val_pop, var_pop)
    show_plots(u_s_10, 10, exp_val_pop, var_pop)


if __name__ == '__main__':
    main()
