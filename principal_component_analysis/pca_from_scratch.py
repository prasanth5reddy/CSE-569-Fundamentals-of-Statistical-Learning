import numpy as np
import matplotlib.pyplot as plt
from load_mnist import mnist


def pca(x, dim=10):
    # take transpose to rearrange rows and columns and store in a new copy
    x_new = np.copy(x.T)
    # centre the data so that after applying svd we get eigen vectors of covariance matrix
    x_new -= np.mean(x_new, axis=0)
    # apply svd on the data
    u, s, vh = np.linalg.svd(x_new, full_matrices=False)
    # select only the first 10 eigen vectors for PCA transformation
    e_k = vh.T[:, :dim]
    # project the original x onto selected eigen vectors
    x_k = np.matmul(np.transpose(e_k), x_new.T)

    # return projected data and the eigen vectors matix
    return x_k, e_k


def plot_covariance_matrix(x):
    # compute covariance matrix of given data x
    cov_mat = np.cov(x)
    # create a new figure with custom size
    fig = plt.figure(figsize=(7, 7))
    # create a new subplot
    ax = fig.add_subplot()
    # plot covariance matrix
    ax.matshow(cov_mat)
    # show covariance matrix values on top of coloured matrix
    for i in range(cov_mat.shape[1]):
        for j in range(cov_mat.shape[0]):
            c = cov_mat[j][i]
            ax.text(i, j, str(round(c, 4)), va='center', ha='center', fontsize=10)
    plt.show()


def reconstruct_data(x, e):
    # reconstruct data by multiplying pca train data with eigen vectors matrix
    return np.matmul(e, x)


def show_original_and_recon(x, recon_x, no_of_samples=5):
    # create a sub function to display images from each class
    def show_one_class(digit=5, start_index=0):
        # create subplots of size 2 x number of samples to show original and reconstructed 5 images
        fig, axes = plt.subplots(2, no_of_samples)
        # set class title
        fig.suptitle('digit = ' + str(digit))
        for i in range(len(axes)):
            for j in range(len(axes[0])):
                if i == 0:
                    # show original images
                    axes[i][j].imshow(x[:, j + start_index].reshape(28, -1), cmap='gray')
                else:
                    # show reconstructed images
                    axes[i][j].imshow(recon_x[:, j + start_index].reshape(28, -1), cmap='gray')
                axes[i][j].axis('off')
        plt.show()

    # show original and reconstructed images for digit 5
    show_one_class(digit=5, start_index=0)
    # show original and reconstructed images for digit 8
    show_one_class(digit=8, start_index=200)


def main():
    # load the data from external load_mnist.py script
    train_X, train_y, test_X, test_y = mnist(noTrSamples=400, noTsSamples=100, digit_range=[5, 8], noTrPerClass=200,
                                             noTsPerClass=50)
    # perform PCA on test and train data
    # Also get the eigen vectors for reconstruction purpose
    pca_train_X, e_pca_train_X = pca(train_X, dim=10)
    pca_test_X, e_pca_test_X = pca(test_X, dim=10)
    # plot the covariance matrix of PCA transformed train data
    plot_covariance_matrix(pca_train_X)

    # reconstruct images from PCA transformed train data and eigen vectors matrix
    recon_train_X = reconstruct_data(pca_train_X, e_pca_train_X)
    # show the reconstructed image samples
    show_original_and_recon(train_X, recon_train_X, no_of_samples=5)


if __name__ == '__main__':
    main()
