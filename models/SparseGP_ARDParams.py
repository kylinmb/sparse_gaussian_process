import tensorflow as tf
import numpy as np
import os
# So it doesn't try and run on CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set random seed
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

jitter = 1e-3


class SparseGPARD:
    def __init__(self, X, y, numSparse):
        self.X = X
        self.y = y
        self.N, self.d = X.shape
        self.M = numSparse

        # input data
        self.tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d])
        self.tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.tf_Xt = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d])  # X*
        self.tf_Xbar = tf.Variable(tf.random.uniform([self.M, self.d]), dtype=tf.float32)  # Pseudo inputs

        # model parameters
        self.tf_log_length_scale = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_amp = tf.constant(0.0, dtype=tf.float32)
        self.tf_log_tau = tf.Variable(0.0, dtype=tf.float32)

        # loss function and optimizer
        self.loss = self.neg_log_likelihood()
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def kernel_matrix(self, x_i, x_j, jitter_dims=None):
        """
        Calculates ARD Kernel Matrix between x_i and x_j.
        :return:K(X, X)
        """
        sq_dist = (tf.reshape(tf.reduce_sum(x_i * x_i, 1), [-1, 1])
                   + tf.reduce_sum(x_j * x_j, 1)
                   - 2 * tf.matmul(x_i, tf.transpose(x_j)))
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * sq_dist)
        if jitter_dims is not None:
            k += jitter * tf.eye(num_rows=jitter_dims[0], num_columns=jitter_dims[1])
        return k

    def bigLambda(self, k_m, k_mn):
        l = tf.transpose(k_mn) * tf.transpose(tf.linalg.solve(k_m, k_mn))
        l = tf.reduce_sum(l, 1)
        l = 1 - l
        l = tf.matrix_diag(l)
        return l

    def Q(self):
        k_m = self.kernel_matrix(self.tf_Xbar, self.tf_Xbar, [self.M, self.M])
        k_mn = self.kernel_matrix(self.tf_Xbar, self.tf_X, [self.M, self.N])
        k_nm = tf.transpose(k_mn)
        sigma2 = 1.0 / tf.exp(self.tf_log_tau) * tf.eye(self.N)
        inverse = tf.linalg.inv(self.bigLambda(k_m, k_mn) + sigma2)
        right = tf.matmul(tf.matmul(k_mn, inverse), k_nm)
        return k_m + right

    def get_full_kernel_matrix(self):
        """
        (K_nm)(K_m^-1)(K_mn) + Lambda + sigma^2I
        :return:
        """
        k_nm = self.kernel_matrix(self.tf_X, self.tf_Xbar, [self.N, self.M])
        k_m = self.kernel_matrix(self.tf_Xbar, self.tf_Xbar, [self.M, self.M])
        k_mn = tf.transpose(k_nm)
        sigma2 = 1.0 / tf.exp(self.tf_log_tau) * tf.eye(self.N)
        k = tf.matmul(tf.matmul(k_nm, tf.linalg.inv(k_m)), k_mn) + self.bigLambda(k_m, k_mn) + sigma2
        return self.sess.run(k, {self.tf_X: self.X, self.tf_y: self.y})

    def predictive_posterior(self):
        """
        mu = (k_*) (Q_m^-1) (K_mn) (Lambda + sigma^2I) (y)
        sigma = K_** - (k_*^T)(K_m^-1 - Q_m^-1)(k_*)
        :return: currently only the predictive mean
        """
        # Mean (mu)
        # (k_ * ^ T)(Q_m ^ -1)
        k_star = self.kernel_matrix(self.tf_Xbar, self.tf_Xt)  # m x r
        q_inv = tf.linalg.inv(self.Q())  # m x m
        first_term = tf.matmul(tf.transpose(k_star), q_inv)  # r X m

        # (K_mn) (Lambda + sigma^2I)
        k_m = self.kernel_matrix(self.tf_Xbar, self.tf_Xbar, [self.M, self.M])
        k_mn = self.kernel_matrix(self.tf_Xbar, self.tf_X, [self.M, self.N])
        sigma2 = 1.0 / tf.exp(self.tf_log_tau)
        lambda_inv = tf.linalg.inv(self.bigLambda(k_m, k_mn) + sigma2 * tf.eye(self.N))
        second_term = tf.matmul(k_mn, lambda_inv)

        # (k_ * ^ T)(Q_m ^ -1)(K_mn)(Lambda + sigma ^ 2I)
        third_term = tf.matmul(first_term, second_term)

        # mu = (k_*^T) (Q_m^-1) (K_mn) (Lambda + sigma^2I) (y)
        predicted_mean = tf.matmul(third_term, self.tf_y)

        # Standard Deviation (sigma)
        k_star_star = self.kernel_matrix(self.tf_Xt, self.tf_Xt)

        predicted_variance = k_star_star - tf.matmul(tf.matmul(tf.transpose(k_star), tf.linalg.inv(k_m) - q_inv), k_star) +  sigma2
        return predicted_mean, predicted_variance

    def neg_log_likelihood(self):
        k_mn = self.kernel_matrix(self.tf_Xbar, self.tf_X, [self.M, self.N])
        k_m = self.kernel_matrix(self.tf_Xbar, self.tf_Xbar, [self.M, self.M])
        m, _ = k_m.shape
        kernels = tf.matmul(tf.transpose(k_mn), tf.linalg.solve(k_m, k_mn))
        sigma2 = 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        S = kernels + self.bigLambda(k_m, k_mn) + sigma2
        L = 0.5*tf.linalg.logdet(S) + 0.5*tf.matmul(tf.transpose(self.tf_y), tf.linalg.solve(S, self.tf_y))[0, 0]
        return L

    def print_loss(self, loss):
        print('Loss: ', loss)

    def train(self):
        tf_dict = {self.tf_X: self.X, self.tf_y: self.y}
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.print_loss)
        print('tau = %g, length-scale = %g' % (
        np.exp(self.tf_log_tau.eval(session=self.sess)), np.exp(self.tf_log_length_scale.eval(session=self.sess))))

    def predict(self, x_star):
        predicted_mean, predicted_variance = self.predictive_posterior()
        return self.sess.run([predicted_mean, predicted_variance], {self.tf_Xt: x_star, self.tf_X: self.X, self.tf_y: self.y})