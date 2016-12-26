from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf


IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 3
NUM_IMAGES = 40160
NUM_CLASSES = 5020
TRAIN_IMAGES = int(NUM_IMAGES * 0.8)
VALID_IMAGES = int(NUM_IMAGES * 0.2)
BATCH_SIZE = 64
MIN_AFTER_DEQUEUE = int(0.4 * TRAIN_IMAGES)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized=serialized, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS])
    reshape = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])
    label = features['label']
    return reshape, label


def input_producer(filename):
    filename_queue = tf.train.string_input_producer([filename])
    image, label = read_and_decode(filename_queue=filename_queue)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=BATCH_SIZE,
                                            capacity=MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE,
                                            min_after_dequeue=MIN_AFTER_DEQUEUE)
    return images, labels


class Inputs(object):
    def __init__(self, train=True):
        self.batch_size = BATCH_SIZE
        self.image_width = IMAGE_WIDTH
        self.image_height = IMAGE_HEIGHT
        self.image_channels = IMAGE_CHANNELS
        self.num_classes = NUM_CLASSES
        self.learning_rate = 1e-4
        if train is True:
            self.images, self.labels = input_producer("records/train.tfrecords")
        else:
            self.images, self.labels = input_producer("records/valid.tfrecords")