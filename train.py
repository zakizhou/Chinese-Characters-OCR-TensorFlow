from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
from model import Recoginizer
from input import Inputs


def main():
    inputs = Inputs(train=True)
    valid_inputs = Inputs(train=False)
    with tf.name_scope("train"):
        with tf.variable_scope("model", reuse=None):
            model = Recoginizer(inputs=inputs)
    with tf.name_scope("valid"):
        with tf.variable_scope("model", reuse=True):
            vmodel = Recoginizer(inputs=valid_inputs)

    sess = tf.Session()
    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        index = 1
        while not coord.should_stop():
            _, loss_value = sess.run([model.train_op, model.loss])
            print("step: " + str(index) + " loss:" + str(loss_value))
            index += 1
            if index % 5 == 0:
                accuracy = sess.run(vmodel.validate)
                print("accuracy is %f" % accuracy)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    except KeyboardInterrupt:
        print("interrupt")
        del sess
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    sess.close()


if __name__ == "__main__":
    main()

