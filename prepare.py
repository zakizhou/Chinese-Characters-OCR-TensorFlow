from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
from scipy.ndimage import imread
import re
import sys
import time
import random
from input import TRAIN_IMAGES


def extract_label_and_index(filename):
    label = int(re.findall(r"(.*?)_", filename)[0])
    index = int(re.findall(r"_(.*?)\.", filename)[0])
    return label, index


def arr2str(arr):
    return arr.tostring()


def convert_to_records(filenames, train=True):
    num_files = len(filenames)
    if train is True:
        print("deal with training data:")
        records_name = "records/train.tfrecords"
    else:
        print("deal with valid data:")
        records_name = "records/valid.tfrecords"
    writer = tf.python_io.TFRecordWriter(records_name)
    for i, filename in enumerate(filenames):
        sys.stdout.write("\r")
        sys.stdout.write("process %6d %% %6d file" % (i + 1, num_files))
        sys.stdout.flush()
        # time.sleep(0.5)
        filepath = os.path.join("/home/windows98/Downloads/zitu", filename)
        label, _ = extract_label_and_index(filename=filename)
        image = imread(filepath)
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr2str(image)])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main():
    filenames = os.listdir("/home/windows98/Downloads/zitu")
    random.shuffle(filenames)
    filenames4train = filenames[:TRAIN_IMAGES]
    filenames4valid = filenames[TRAIN_IMAGES+1:]
    convert_to_records(filenames4train, True)
    convert_to_records(filenames4valid, False)


if __name__ == "__main__":
    main()