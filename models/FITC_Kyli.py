import tensorflow as tf
import numpy as np

# Set random seed TODO why?
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

noise = 1e-3


class FITC:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N, self.d = X.shape
        # input data
        self.tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d])
        self.tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.tf_Xt = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d])
        # model parameters
        self.tf_Xbar = tf.Variable(tf.zeros([10, self.d]), dtype=tf.float32)  #TODO what is the right number for M
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

    def kernel_matrix_M(self):
        col_norm2 = tf.reduce_sum(self.tf_Xbar * self.tf_Xbar, 1)
        col_norm2 = tf.reshape(col_norm2, [-1, 1])
        k = col_norm2 - 2.0 * tf.matmul(self.tf_Xbar, tf.transpose(self.tf_Xbar)) + tf.transpose(col_norm2)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def kernel_matrix_NM(self):
        col_norm_bar = tf.reduce_sum(self.tf_Xbar * self.tf_Xbar, 1)
        col_norm_bar = tf.reshape(col_norm_bar, [-1, 1])
        col_norm_n = tf.reduce_sum(self.tf_X * self.tf_X, 1)
        col_norm_n = tf.reshape(col_norm_n, [-1, 1])
        k = col_norm_bar + col_norm_n - 2.0 * tf.matmul(self.tf_X, tf.transpose(self.tf_Xbar))
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def neg_log_likelihood(self):
        Knm = self.kernel_matrix_NM()
        Km = self.kernel_matrix_M()
        Lambda = 1 - tf.reduce_sum(tf.math.multiply(tf.transpose(Knm), tf.transpose(tf.linalg.solve(Km, Knm))), 1)
        Lambda = tf.linalg.diag(Lambda)
        S = Knm*tf.linalg.solve(Km, tf.transpose(Knm)) + Lambda + 1.0/tf.exp(self.tf_log_tau)*tf.eye(10) # TODO not sure if the should be N or M
        L = 0.5*tf.linalg.logdet(S) + 0.5*tf.matmul(tf.transpose(self.tf_y), tf.linalg.solve(S, self.tf_y))[0, 0]
        return L

    def print_loss(self, loss):
        print('Loss: ', loss)

    def train(self):
        tf_dict = {self.tf_X: self.X, self.tf_y: self.y}
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.print_loss)
        print('tau = %g, length-scale = %g' % (
        np.exp(self.tf_log_tau.eval(session=self.sess)), np.exp(self.tf_log_length_scale.eval(session=self.sess))))