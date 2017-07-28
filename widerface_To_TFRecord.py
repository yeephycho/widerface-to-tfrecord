from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
import cv2
import os

from utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', 'output/train.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def parse_example(f):
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = b'jpg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)


  filename = f.readline().rstrip()
  print(filename)
  filepath = os.path.join("./WIDER/WIDER_train/images/", filename)
  print(filepath)
  image_raw = cv2.imread(filepath)

  encoded_image_data = open(filepath).read()
  height, width, channel = image_raw.shape
  print("height is %d, width is %d, channel is %d" % (height, width, channel))

  face_num = int(f.readline().rstrip())
  print(face_num)

  for i in range(face_num):
    annot = f.readline().rstrip().split()
    # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
    xmins.append( max(0, (float(annot[0]) / width) ) )
    ymins.append( max(0, (float(annot[1]) / height) ) )
    xmaxs.append( min(1, ((float(annot[0]) + float(annot[2])) / width) ) )
    ymaxs.append( min(1, ((float(annot[1]) + float(annot[3])) / height) ) )
    classes_text.append('face')
    classes.append(1)
    print(xmins[i], ymins[i], xmaxs[i], ymaxs[i], classes_text[i], classes[i])


  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(int(height)),
    'image/width': dataset_util.int64_feature(int(width)),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

  return tf_example


def main(unused_argv):
  f = open("WIDER/wider_face_train_annot.txt")
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

# WIDER FACE DATASET ANNOTATED 12880 IMAGES
  for image_idx in range(12880):
    print(image_idx)
    tf_example = parse_example(f)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
