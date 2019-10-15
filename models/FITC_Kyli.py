import tensorflow as tf
import numpy as np

# Set random seed TODO why?
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

jitter = 1e-3


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
        self.tf_Xbar = tf.Variable(tf.random.uniform([40, self.d]), dtype=tf.float32)  #TODO what is the right number for n
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
        # rows, columns = k.shape
        # k = jitter * np.eye(rows, columns)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def get_kernel_matrix(self):
        k = self.kernel_matrix_M()
        return self.sess.run(k, {self.tf_X: self.X, self.tf_y: self.y})

    def kernel_matrix_MN(self):
        col_norm_bar = tf.reduce_sum(self.tf_Xbar * self.tf_Xbar, 1)
        col_norm_bar = tf.reshape(col_norm_bar, [-1, 1])
        col_norm_n = tf.reduce_sum(self.tf_X * self.tf_X, 1)
        col_norm_n = tf.transpose(tf.reshape(col_norm_n, [-1, 1]))
        k = col_norm_bar + col_norm_n - 2.0 * tf.matmul(self.tf_Xbar, tf.transpose(self.tf_X))
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def bigLambda(self, km, kmn):
        l = tf.transpose(kmn) * tf.transpose(tf.matmul(tf.matrix_inverse(km), kmn))
        l = tf.reduce_sum(l, 1)
        l = 1 - l
        l = tf.matrix_diag(l)
        return l

    def kernel_cross(self):
        col_norm1 = tf.reshape(tf.reduce_sum(self.tf_Xt * self.tf_Xt, 1), [-1, 1])
        col_norm2 = tf.reshape(tf.reduce_sum(self.tf_Xbar * self.tf_Xbar, 1), [-1, 1])
        k = col_norm1 - 2.0 * tf.matmul(self.tf_Xt, tf.transpose(self.tf_Xbar)) + tf.transpose(col_norm2)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0 / tf.exp(self.tf_log_length_scale) * k)
        return k

    def kernel_cross_star(self):
        col_norm2 = tf.reduce_sum(self.tf_Xt * self.tf_Xt, 1)
        col_norm2 = tf.reshape(col_norm2, [-1, 1])
        k = col_norm2 - 2.0 * tf.matmul(self.tf_Xt, tf.transpose(self.tf_Xt)) + tf.transpose(col_norm2)
        # rows, columns = k.shape
        # k = jitter * np.eye(rows, columns)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def predict(self):
        KM = self.kernel_matrix_M()
        KMN = self.kernel_matrix_MN()
        sigma2 = 1.0 / tf.exp(self.tf_log_tau) * tf.eye(self.N)
        lam = tf.matrix_inverse(self.bigLambda(KM, KMN) + sigma2)
        Q = KM + tf.matmul(tf.matmul(KMN, lam), tf.transpose(KMN))
        Q = tf.matrix_inverse(Q)
        kstarm = self.kernel_cross()
        predicted_mean = tf.matmul(tf.matmul(tf.transpose(kstarm), Q), KMN)
        predicted_mean = tf.matmul(predicted_mean, lam)
        predicted_mean = tf.matmul(predicted_mean, self.tf_y)
        predicted_variance = tf.matmul(tf.matmul(tf.transpose(kstarm), tf.matrix_inverse(KM) - Q), kstarm)
        predicted_variance = self.kernel_cross_star() - predicted_variance + 1.0 / tf.exp(self.tf_log_tau)
        return predicted_mean, predicted_variance

    def eval(self, xStar):
        predicted_mean, predicted_variance = self.predict()
        return self.sess.run([predicted_mean, predicted_variance], {self.tf_Xt: xStar, self.tf_X: self.X, self.tf_y: self.y})

    def neg_log_likelihood(self):
        Kmn = self.kernel_matrix_MN()
        Km = self.kernel_matrix_M()
        kernels = tf.matmul(tf.transpose(Kmn), tf.linalg.solve(Km, Kmn))
        sigma2 = 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        S = kernels + self.bigLambda(Km, Kmn) + sigma2
        L = 0.5*tf.linalg.logdet(S) + 0.5*tf.matmul(tf.transpose(self.tf_y), tf.linalg.solve(S, self.tf_y))[0, 0]
        return L

    def print_loss(self, loss):
        print('Loss: ', loss)

    def train(self):
        tf_dict = {self.tf_X: self.X, self.tf_y: self.y}
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.print_loss)
        print('tau = %g, length-scale = %g' % (
        np.exp(self.tf_log_tau.eval(session=self.sess)), np.exp(self.tf_log_length_scale.eval(session=self.sess))))