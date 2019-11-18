import tensorflow as tf
import numpy as np

# Set random seed TODO why?
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

jitter = 1e-3


class FITC:
    def __init__(self, X, y, numSparse):
        self.X = X
        self.y = y
        self.N, self.d = X.shape
        # input data
        self.tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d])
        self.tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.tf_Xt = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d])
        # model parameters
        self.tf_Xbar = tf.Variable(tf.random.uniform([numSparse, self.d]), dtype=tf.float32)  #TODO what is the right number for n
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

    def KM(self):
        col_norm2 = tf.reduce_sum(self.tf_Xbar * self.tf_Xbar, 1)
        col_norm2 = tf.reshape(col_norm2, [-1, 1])
        k = col_norm2 - 2.0 * tf.matmul(self.tf_Xbar, tf.transpose(self.tf_Xbar)) + tf.transpose(col_norm2)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def KMN(self):
        col_norm_bar = tf.reduce_sum(self.tf_Xbar * self.tf_Xbar, 1)
        col_norm_bar = tf.reshape(col_norm_bar, [-1, 1])
        col_norm_n = tf.reduce_sum(self.tf_X * self.tf_X, 1)
        col_norm_n = tf.transpose(tf.reshape(col_norm_n, [-1, 1]))
        k = col_norm_bar + col_norm_n - 2.0 * tf.matmul(self.tf_Xbar, tf.transpose(self.tf_X))
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def kernel_cross(self):
        col_norm1 = tf.reshape(tf.reduce_sum(self.tf_Xt * self.tf_Xt, 1), [-1, 1])
        col_norm2 = tf.reshape(tf.reduce_sum(self.tf_Xbar * self.tf_Xbar, 1), [-1, 1])
        k = col_norm1 - 2.0 * tf.matmul(self.tf_Xt, tf.transpose(self.tf_Xbar)) + tf.transpose(col_norm2)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def kernel_cross_star(self):
        col_norm2 = tf.reduce_sum(self.tf_Xt * self.tf_Xt, 1)
        col_norm2 = tf.reshape(col_norm2, [-1, 1])
        k = col_norm2 - 2.0 * tf.matmul(self.tf_Xt, tf.transpose(self.tf_Xt)) + tf.transpose(col_norm2)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def bigLambda(self, KM, KMN):
        l = tf.transpose(KMN) * tf.transpose(tf.matmul(tf.matrix_inverse(KM), KMN))
        l = tf.reduce_sum(l, 1)
        l = 1 - l
        l = tf.matrix_diag(l)
        return l

    def get_kernel_matrix(self):
        k = self.KM()
        return self.sess.run(k, {self.tf_X: self.X, self.tf_y: self.y})

    def Q(self):
        KM = self.KM()
        KMN = self.KMN()
        KNM = tf.transpose(KMN)
        sigma2 = 1.0 / tf.exp(self.tf_log_tau) * tf.eye(self.N)
        inverse = tf.matrix_inverse(self.bigLambda(KM, KMN) + sigma2)
        right = tf.matmul(tf.matmul(KMN, inverse), KNM)
        Q = KM + right
        return Q

    def predict_mean(self):
        KM = self.KM()
        KMN = self.KMN()

        KSTAR = tf.transpose(self.kernel_cross())
        QINV = tf.matrix_inverse(self.Q())
        first_mult = tf.matmul(KSTAR, QINV)

        sigma2 = 1.0 / tf.exp(self.tf_log_tau) * tf.eye(self.N)
        LAMBDAINV = tf.matrix_inverse(self.bigLambda(KM, KMN) + sigma2)
        second_mult = tf.matmul(KMN, LAMBDAINV)

        third_mult = tf.matmul(first_mult, second_mult)
        print(tf.shape(third_mult))
        return tf.matmul(third_mult, self.tf_y)

    def predict(self):
        return self.predict_mean()
    # def predict(self):
    #     KM = self.KM()
    #     KMN = self.KMN()
    #     sigma2 = 1.0 / tf.exp(self.tf_log_tau) * tf.eye(self.N)
    #     lam = tf.matrix_inverse(self.bigLambda(KM, KMN) + sigma2)
    #     Q = KM + tf.matmul(tf.matmul(KMN, lam), tf.transpose(KMN))
    #     Q = tf.matrix_inverse(Q)
    #     kstarm = self.kernel_cross()
    #     predicted_mean = tf.matmul(tf.matmul(kstarm, Q), KMN)
    #     predicted_mean = tf.matmul(predicted_mean, lam)
    #     predicted_mean = tf.matmul(predicted_mean, self.tf_y)
    #     predicted_variance = tf.matmul(tf.matmul(kstarm, tf.matrix_inverse(KM) - Q), tf.transpose(kstarm))
    #     predicted_variance = self.kernel_cross_star() - predicted_variance + 1.0 / tf.exp(self.tf_log_tau)
    #     return predicted_mean, predicted_variance

    def eval(self, xStar):
        predicted_mean = self.predict()
        return self.sess.run([predicted_mean], {self.tf_Xt: xStar, self.tf_X: self.X, self.tf_y: self.y})

    # def eval(self, xStar):
    #     predicted_mean, predicted_variance = self.predict()
    #     return self.sess.run([predicted_mean, predicted_variance], {self.tf_Xt: xStar, self.tf_X: self.X, self.tf_y: self.y})

    def neg_log_likelihood(self):
        KMN = self.KMN()
        KM = self.KM()
        kernels = tf.matmul(tf.transpose(KMN), tf.linalg.solve(KM, KMN))
        sigma2 = 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        S = kernels + self.bigLambda(KM, KMN) + sigma2
        L = 0.5*tf.linalg.logdet(S) + 0.5*tf.matmul(tf.transpose(self.tf_y), tf.linalg.solve(S, self.tf_y))[0, 0]
        return L

    def print_loss(self, loss):
        print('Loss: ', loss)

    def train(self):
        tf_dict = {self.tf_X: self.X, self.tf_y: self.y}
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.print_loss)
        print('tau = %g, length-scale = %g' % (
        np.exp(self.tf_log_tau.eval(session=self.sess)), np.exp(self.tf_log_length_scale.eval(session=self.sess))))