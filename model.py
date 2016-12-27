from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf


class Recoginizer(object):
    def __init__(self, inputs):
        batch_size = inputs.batch_size
        images = tf.cast(inputs.images, tf.float32)
        labels = inputs.labels
        num_classes = inputs.num_classes
        learning_rate = inputs.learning_rate
        unigrams = inputs.unigrams
        num_sampled = inputs.num_sampled
        with tf.variable_scope("conv_pool_1"):
            filter = tf.get_variable(name="filter",
                                     shape=[5, 5, 3, 64],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05),
                                     dtype=tf.float32)
            bias = tf.get_variable(name="bias",
                                   shape=[64],
                                   initializer=tf.constant_initializer(value=0.01),
                                   dtype=tf.float32)
            conv = tf.nn.conv2d(input=images,
                                filter=filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name="conv")
            relu = tf.nn.relu(tf.add(conv, bias), name="relu")
            pool = tf.nn.max_pool(value=relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME")
        with tf.variable_scope("conv_pool_2"):
            filter = tf.get_variable(name="filter",
                                     shape=[5, 5, 64, 128],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05),
                                     dtype=tf.float32)
            bias = tf.get_variable(name="bias",
                                   shape=[128],
                                   initializer=tf.constant_initializer(value=0.01),
                                   dtype=tf.float32)
            conv = tf.nn.conv2d(input=pool,
                                filter=filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name="conv")
            relu = tf.nn.relu(tf.add(conv, bias), name="relu")
            pool = tf.nn.max_pool(value=relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME")
        reshape = tf.reshape(pool, shape=[batch_size, -1], name="reshape")
        dim = reshape.get_shape().as_list()[-1]
        with tf.variable_scope("fully_conn"):
            weights = tf.get_variable(name="weights",
                                      shape=[dim, 192],
                                      initializer=tf.truncated_normal_initializer(stddev=0.05),
                                      dtype=tf.float32)
            bias = tf.get_variable(name="bias",
                                   shape=[192],
                                   initializer=tf.constant_initializer(value=0.01),
                                   dtype=tf.float32)
            xw_plus_b = tf.nn.xw_plus_b(reshape, weights, bias)
            relu = tf.nn.relu(xw_plus_b)
        with tf.variable_scope("sampled_loss"):
            softmax_w = tf.get_variable(name="softmax_w",
                                        shape=[num_classes, 192],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable(name="softmax_b",
                                        shape=[num_classes],
                                        initializer=tf.constant_initializer(value=0.01),
                                        dtype=tf.float32)
            labeled_weights = tf.nn.embedding_lookup(softmax_w, labels)
            labeled_bias = tf.nn.embedding_lookup(softmax_b, labels)
            sampled_ids = tf.nn.fixed_unigram_candidate_sampler(true_classes=labels,
                                                                num_true=1,
                                                                num_sampled=num_sampled,
                                                                unique=True,
                                                                unigrams=unigrams,
                                                                distortion=0.75)
            sampled_weights = tf.nn.embedding_lookup(softmax_w, sampled_ids)
            sampled_bias = tf.nn.embedding_lookup(softmax_b, sampled_ids)
            # [batch_size]
            labeld_logits = tf.reduce_sum(tf.mul(relu, labeled_weights), 1) + labeled_bias
            # [batch_size, num_sampled]
            sampled_logits = tf.matmul(relu, sampled_weights, transpose_b=True) + sampled_bias
            logits = tf.concat(1, [tf.expand_dims(labeld_logits, -1), sampled_logits])
            self.__logits = tf.nn.xw_plus_b(relu, tf.transpose(softmax_w), softmax_b)
        with tf.name_scope("loss"):
            self.__loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.__train_op = optimizer.minimize(self.__loss, name="train_op")
        with tf.name_scope("valid"):
            predict = tf.argmax(self.__logits, 1)
            equal = tf.cast(tf.equal(predict, labels), tf.float32)
            self.__accuracy = tf.reduce_mean(equal)


        # with tf.variable_scope("softmax"):
        #     softmax_w = tf.get_variable(name="softmax_w",
        #                                 shape=[192, num_classes],
        #                                 initializer=tf.truncated_normal_initializer(stddev=0.05),
        #                                 dtype=tf.float32)
        #     softmax_b = tf.get_variable(name="softmax_b",
        #                                 shape=[num_classes],
        #                                 initializer=tf.constant_initializer(value=0.01),
        #                                 dtype=tf.float32)
        #     logits = tf.nn.xw_plus_b(relu, softmax_w, softmax_b)
        # with tf.name_scope("loss"):
        #     loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        #     self.__loss = tf.reduce_mean(loss_per_example, name="loss")
        # with tf.name_scope("train"):
        #     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #     self.__train_op = optimizer.minimize(self.__loss, name="train_op")
        # with tf.name_scope("valid"):
        #     predict = tf.argmax(logits, 1)
        #     equal = tf.cast(tf.equal(predict, labels), tf.float32)
        #     self.__accuracy = tf.reduce_mean(equal)


    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op

    @property
    def validate(self):
        return self.__accuracy

