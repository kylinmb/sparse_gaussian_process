import tensorflow as tf
import numpy as np

# Set random seed TODO why?
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

noise = 1e-3


class GPRK:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N, self.d = X.shape
        self.tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d])  # None = first dim can be any size
        self.tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.tf_Xt = tf.compat.v1.placeholder(tf.float32, shape=[None, self.d]) # Is this the test data?
        # model parameters
        self.tf_log_length_scale = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_amp = tf.Variable(0.0, dtype=tf.float32) # TODO should this be constant or varaible
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

    def kernel_matrix(self):
        col_norm2 = tf.reduce_sum(self.tf_X * self.tf_X, 1)
        col_norm2 = tf.reshape(col_norm2, [-1, 1])  # TODO why the reshape?
        k = col_norm2 - 2.0 * tf.matmul(self.tf_X, tf.transpose(self.tf_X)) + tf.transpose(col_norm2)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def get_kernel_matrix(self):
        k = self.kernel_matrix()
        return self.sess.run(k, {self.tf_X: self.X, self.tf_y: self.y})

    def kernel_cross(self):
        col_norm1 = tf.reshape(tf.reduce_sum(self.tf_Xt * self.tf_Xt, 1), [-1, 1])
        col_norm2 = tf.reshape(tf.reduce_sum(self.tf_X * self.tf_X, 1), [-1, 1])
        k = col_norm1 - 2.0 * tf.matmul(self.tf_Xt, tf.transpose(self.tf_X)) + tf.transpose(col_norm2)
        k = tf.exp(self.tf_log_amp) * tf.exp(-1.0/tf.exp(self.tf_log_length_scale) * k)
        return k

    def predict(self):
        s = self.kernel_matrix() + 1.0 / tf.exp(self.tf_log_tau) * tf.eye(self.N)
        kmn = self.kernel_cross()
        predicted_mean = tf.matmul(kmn, tf.linalg.solve(s, self.tf_y))
        predicted_variance = tf.exp(self.tf_log_amp) + noise - tf.reduce_sum(kmn * tf.transpose(tf.linalg.solve(s, tf.transpose(kmn))), 1)
        return predicted_mean, predicted_variance

    def eval(self, xStar):
        predicted_mean, predicted_variance = self.predict()
        return self.sess.run([predicted_mean, predicted_variance], {self.tf_Xt: xStar, self.tf_X: self.X, self.tf_y: self.y})

    def neg_log_likelihood(self):
        S = self.kernel_matrix() + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        L = 0.5*tf.linalg.logdet(S) + 0.5*tf.matmul(tf.transpose(self.tf_y), tf.linalg.solve(S, self.tf_y))[0, 0]
        return L

    def print_loss(self, loss):
        print('Loss: ', loss)

    def train(self):
        tf_dict = {self.tf_X: self.X, self.tf_y: self.y}
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.print_loss)
        print('tau = %g, length-scale = %g' % (np.exp(self.tf_log_tau.eval(session=self.sess)), np.exp(self.tf_log_length_scale.eval(session=self.sess))))

