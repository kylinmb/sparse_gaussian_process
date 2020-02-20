import tensorflow as tf
import numpy as np
import os

# So it doesn't try and run on CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set random seed
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

jitter = 1e-3


def print_loss(loss):
    print('Loss: ', loss)


class FullGPARD:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N, self.d = X.shape
        self.tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d])  # None = first dim can be any size
        self.tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.tf_Xt = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d]) # xStar
        # model parameters
        self.tf_log_length_scale = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_amp = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_tau = tf.Variable(0.0, dtype=tf.float32)
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
        sq_dist = (tf.reshape(tf.reduce_sum(x_i ** 2, 1), [-1, 1])
                   + tf.reduce_sum(x_j ** 2, 1)
                   - 2 * tf.matmul(x_i, tf.transpose(x_j)))
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * sq_dist)
        if jitter_dims is not None:
            k += jitter * tf.eye(jitter_dims[0], jitter_dims[1])
        return k

    def get_kernel_matrix(self):
        k = self.kernel_matrix(self.tf_X, self.tf_X, [self.N, self.N])
        return self.sess.run(k, {self.tf_X: self.X})

    def predictive_posterior(self):
        # k_nn + (sigma^2)(I)
        sigma = 1.0 / tf.exp(self.tf_log_tau)
        s = self.kernel_matrix(self.tf_X, self.tf_X, [self.N, self.N]) + sigma * tf.eye(self.N)
        # k_nr k(X, x*)
        k_nr = self.kernel_matrix(self.tf_X, self.tf_Xt)
        predicted_mean = tf.matmul(tf.transpose(k_nr), tf.linalg.solve(s, self.tf_y))
        # TODO the commented code is from following Shandian's code, not sure why the jitter was added
        # predicted_variance = (tf.exp(self.tf_log_amp)
        #                       - tf.reduce_sum(k_rn * tf.transpose(tf.linalg.solve(s, tf.transpose(k_rn))), 1)
        #                       + jitter)  # Jitter for numerical stability
        predicted_variance = (self.kernel_matrix(self.tf_Xt, self.tf_Xt)
                              - tf.matmul(tf.transpose(k_nr), tf.linalg.solve(s, k_nr))
                              + sigma)
        return predicted_mean, predicted_variance

    def predict(self, xStar):
        predicted_mean, predicted_variance = self.predictive_posterior()
        return self.sess.run([predicted_mean, predicted_variance], {self.tf_Xt: xStar, self.tf_X: self.X, self.tf_y: self.y})

    def neg_log_likelihood(self):
        S = self.kernel_matrix(self.tf_X, self.tf_X, [self.N, self.N]) + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        L = 0.5*tf.linalg.logdet(S) + 0.5*tf.matmul(tf.transpose(self.tf_y), tf.linalg.solve(S, self.tf_y))[0, 0]
        return L

    def train(self):
        tf_dict = {self.tf_X: self.X, self.tf_y: self.y}
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=print_loss)
        print('tau = %g, length-scale = %g' % (np.exp(self.tf_log_tau.eval(session=self.sess)), np.exp(self.tf_log_length_scale.eval(session=self.sess))))

