import tensorflow as tf
import numpy
import cv2

def read_and_decode():
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/source_id': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/format': tf.FixedLenFeature([], tf.string),
                'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
                'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
                'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
                'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
                'image/object/class/text': tf.FixedLenFeature([], tf.string),
                'image/object/class/label': tf.FixedLenFeature([], tf.int64)
                }
            )
    image = tf.image.decode_jpeg(features['image/encoded'])
    label = tf.cast(features['image/object/class/label'], tf.int32)
    xmin = features['image/object/bbox/xmin']
    xmax = features['image/object/bbox/xmax']
    ymin = features['image/object/bbox/ymin']
    ymax = features['image/object/bbox/ymax']
    return image, label, xmin, xmax

filename = "output/train.tfrecord"
filename_queue = tf.train.string_input_producer([filename])


with tf.Session() as sess:
    image, label, xmin, xmax = read_and_decode()
   # image = tf.reshape(image, [1200, 1032, 3])
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1):
        image_out, label_out, xmin_out, xmax_out = sess.run([image, label, xmin, xmax])
        print(image_out.shape)
        print(label_out)
        print(xmin_out)
        print(xmax_out)

    coord.request_stop()
    coord.join(threads)

