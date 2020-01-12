import numpy as np
import matplotlib.pyplot as plt
from load_mnist import mnist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from principal_component_analysis.pca_from_scratch import pca


class FLD:
    def __init__(self, x, y, dim=1):
        self.x = x
        self.y = y
        self.dim = dim
        self.c_1_mean = None
        self.c_2_mean = None
        self.e_k = None
        self.thresh = None
        self.is_c1_less_than_thresh = None

    def fld(self):
        # split the train data to two classes
        c_1 = self.x[:, :200]
        c_2 = self.x[:, 200:]
        # calculate the mean vector for each class train data
        self.c_1_mean = np.mean(c_1, axis=1).reshape((self.x.shape[0], 1))
        self.c_2_mean = np.mean(c_2, axis=1).reshape((self.x.shape[0], 1))

        # calculate the between-class covariance matrix
        s_b = np.dot(self.c_2_mean - self.c_1_mean, np.transpose(self.c_2_mean - self.c_1_mean))

        # s_b_1 = 200 * np.dot(self.c_1_mean - x_mean, np.transpose(self.c_1_mean - x_mean))
        # s_b_2 = 200 * np.dot(self.c_2_mean - x_mean, np.transpose(self.c_2_mean - x_mean))
        # s_b = s_b_1 + s_b_2

        # calculate the within-class covariance matrix
        s_w = np.cov(c_1) + np.cov(c_2)

        # compute eigen values and eigen vectors for inverse(s_w) * s_b
        e_vals, e_vecs = np.linalg.eig(np.dot(np.linalg.pinv(s_w), s_b))
        # sorted the eigen values in decreasing order and return their indices
        e_vals_sorted_ind = np.argsort(e_vals)[::-1]
        # get eigen vector corresponding to max eigen value
        e_k = np.real(e_vecs[:, e_vals_sorted_ind[0]]).reshape(-1, 1)
        # store the eigen
        self.e_k = e_k

    def fit(self):
        self.fld()
        # project each class means
        c1_k = np.matmul(np.transpose(self.e_k), self.c_1_mean)
        c2_k = np.matmul(np.transpose(self.e_k), self.c_2_mean)
        # set threshold as mean of projected means
        self.thresh = (c1_k.item() + c2_k.item()) / 2
        # check if class 1 mean is less than threshold and store it for prediction
        self.is_c1_less_than_thresh = True if c1_k < self.thresh else False

    def accuracy(self, x, y):
        # project data onto eigen vector matrix
        e_x = np.matmul(np.transpose(self.e_k), x)
        # initialise predictions
        pred = np.zeros_like(y)
        # iterate through each projected data
        for i in range(e_x.shape[1]):
            # check if class 1 is less than threshold
            if self.is_c1_less_than_thresh:
                # assign 5 if projected data is less than threshold
                pred[:, i] = 5 if e_x[:, i] < self.thresh else 8
            else:
                # assign 8 if projected data is less than threshold
                pred[:, i] = 8 if e_x[:, i] < self.thresh else 5
        # return number of correct predictions / total number of data
        return np.sum(pred == y) / y.shape[1]

    def train_accuracy(self):
        return self.accuracy(self.x, self.y)

    def test_accuracy(self, t_X, t_y):
        return self.accuracy(t_X, t_y)


def main():
    # load the data from external load_mnist.py script
    train_X, train_y, test_X, test_y = mnist(noTrSamples=400, noTsSamples=100, digit_range=[5, 8], noTrPerClass=200,
                                             noTsPerClass=50)
    # get pca data
    pca_train_X, e_pca_train_X = pca(train_X, dim=10)
    pca_test_X, e_pca_test_X = pca(test_X, dim=10)

    # apply Fishers Linear Discriminant to project PCA trained data
    pca_mnist_fld = FLD(pca_train_X, train_y, dim=1)
    # fit the FLD process
    pca_mnist_fld.fit()
    # compute training accuracy
    train_acc = pca_mnist_fld.train_accuracy()
    # compute test accuracy
    test_acc = pca_mnist_fld.test_accuracy(pca_test_X, test_y)
    # Display training and test accuracy
    print(f'Training accuracy : {train_acc}')
    print(f'Test accuracy : {test_acc}')

    clf = LinearDiscriminantAnalysis()
    # print(pca_train_X.T.shape, train_y.T.flatten().shape)  # (400, 10) (400,)
    clf.fit(pca_train_X.T, train_y.T.flatten())

    print('From sklearn')
    print('Training accuracy : ', clf.fit(pca_train_X.T, train_y.T.flatten()).score(pca_train_X.T, train_y.T.flatten()))
    print('Test accuracy : ', clf.score(pca_test_X.T, test_y.T.flatten()))


if __name__ == '__main__':
    main()
