import numpy as np
from sklearn.cluster import KMeans


def get_binomial_20_flips():
    # read given txt file
    with open('./Binomial_20_flips.txt') as file:
        # get each line as element of the list
        lines = file.read().splitlines()
    # convert type from str to int, reshape and return the result
    return np.array(list(map(int, lines))).reshape((-1, 1))


def k_means(points, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
    # KMeans divide the points into three clusters 0, 1, and 2
    no_of_cluster_0s = np.sum(kmeans.labels_ == 0)
    no_of_cluster_1s = np.sum(kmeans.labels_ == 1)
    no_of_cluster_2s = np.sum(kmeans.labels_ == 2)
    no_of_total_points = points.shape[0]

    # compute prior probabilities
    pi_1 = no_of_cluster_0s / no_of_total_points
    pi_2 = no_of_cluster_1s / no_of_total_points
    pi_3 = no_of_cluster_2s / no_of_total_points

    print('Prior Probabilities')
    print('pi_1 : ', pi_1)
    print('pi_2 : ', pi_2)
    print('pi_3 : ', pi_3)

    no_of_heads_0 = np.sum(points[kmeans.labels_ == 0])
    no_of_heads_1 = np.sum(points[kmeans.labels_ == 1])
    no_of_heads_2 = np.sum(points[kmeans.labels_ == 2])

    # compute probability of heads
    theta_1 = no_of_heads_0 / (20 * no_of_cluster_0s)
    theta_2 = no_of_heads_1 / (20 * no_of_cluster_1s)
    theta_3 = no_of_heads_2 / (20 * no_of_cluster_2s)

    print('Parameters (Probability of Heads)')
    print('theta_1 : ', theta_1)
    print('theta_2 : ', theta_2)
    print('theta_3 : ', theta_3)

    return np.array([pi_1, pi_2, pi_3]), np.array([theta_1, theta_2, theta_3])


def main():
    # get points from txt file
    points = get_binomial_20_flips()
    # set k value
    k = 3
    # K means algorithm
    print('K means')
    k_means(points, k)


if __name__ == '__main__':
    main()
