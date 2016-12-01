import tensorflow as tf
import numpy
import scipy.sparse as ssp


OPTIMIZER = {"adam": tf.train.AdamOptimizer,
             "sgd": tf.train.GradientDescentOptimizer}


class LSNMF(object):
    """
    Label-based Semi-supervised NMF
    """

    def __init__(self, K, batch_size=128, lambda_c=1.0, optimizer="adam", learning_rate=1e-2,
                 threads=8, seed=42):
        """
        :param N: the number of nodes
        :param K: dimension of latent vectors
        :param lambda_c: weight of supervised term
        :param optimizer: optimizer
        :param learning_rate: learning rate
        :param threads: the number of threads
        """
        self.K = K
        self.lambda_c = lambda_c
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.threads = threads
        self.seed = seed
        self.batch_size = batch_size

    def setup(self, A, O):
        """
        set up inumpyut matrices

        :param A: adjacency matrix, dense or sparse matrix
        :param O: constraint matrix, dense or sparse matrix
        """
        self.sum_weight = A.sum()
        self.sum_const = O.sum()
        self.A = ssp.lil_matrix(A)
        self.O = ssp.lil_matrix(O)
        self.N = A.shape[0]

        self._build()

    def sample_indices(self):
        """
        sample node indices

        :return: list of unique node indices
        """
        batch_size = min(self.N, self.batch_size)
        return numpy.random.choice(self.N, batch_size, replace=False)

    def partial_fit(self, indices):
        """
        run partial fitting for given node indices

        :param indices: sequence of nose indices
        :return: loss
        """
        _, loss = self.sess.run([self.opt, self.loss],
                                {self.targets: numpy.array(indices, dtype=numpy.int32)})
        return loss

    def _build(self):

        a_nnz = self.A.nonzero()
        a_indices = numpy.array(a_nnz).T
        a_values = self.A[a_nnz].toarray()[0]

        self.a_sum = a_values.sum()
        a_values_norm = numpy.array(a_values / self.a_sum, dtype=numpy.float32)


        # adjacency matrix
        self._A = tf.sparse_to_dense(a_indices, output_shape=[self.N, self.N],
                                     sparse_values=a_values_norm)

        o_nnz = self.O.nonzero()
        o_indices = numpy.array(o_nnz).T
        o_values = self.O[o_nnz].toarray()[0]
        self.o_sum = o_values.sum()
        o_values_norm = numpy.array(o_values / self.o_sum, dtype=numpy.float32)

        # supervised matrix
        self._O = tf.sparse_to_dense(o_indices, output_shape=[self.N, self.N],
                                     sparse_values=o_values_norm)

        self.targets = tf.placeholder(tf.int64, shape=[None], name="targets")


        # latent vectors
        initializer = tf.random_uniform_initializer(0, 1 / numpy.sqrt(self.N * self.N * self.K))
        self._H = tf.get_variable("H", shape=(self.N, self.K),
                                  dtype=tf.float32, initializer=initializer)

        A_target = tf.gather(tf.transpose(tf.gather(self._A, self.targets)), self.targets)
        O_target = tf.gather(tf.transpose(tf.gather(self._O, self.targets)), self.targets)

        H_target = tf.gather(self._H, self.targets)

        Y_target = tf.matmul(H_target, tf.transpose(H_target))

        self.loss = tf.nn.l2_loss(A_target - Y_target)

        self.opt = OPTIMIZER[self.optimizer](self.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    @property
    def H(self):
        return self.sess.run(self._H)


if __name__ == "__main__":
    model = LSNMF(4)

    A = ssp.lil_matrix((10, 10))
    O = ssp.lil_matrix((10, 10))

    A[[0, 1, 2], [1, 0, 3]] = 1.0
    O[[0, 1, 2], [1, 0, 3]] = 1.0


    model.setup(A, O)
    H = model.H

    model.partial_fit([1, 2, 5])
    print(H)

    print(model.sample_indices())
