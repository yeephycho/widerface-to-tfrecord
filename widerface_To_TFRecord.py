from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
import cv2
import os
import hashlib

from utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', 'output/train.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def parse_example(f):
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  poses = []
  truncated = []
  difficult_obj = []



  filename = f.readline().rstrip()
  print(filename)
  filepath = os.path.join("./WIDER/WIDER_train/images/", filename)
  print(filepath)
  image_raw = cv2.imread(filepath)

  encoded_image_data = open(filepath).read()
  key = hashlib.sha256(encoded_image_data).hexdigest()

  height, width, channel = image_raw.shape
  print("height is %d, width is %d, channel is %d" % (height, width, channel))

  face_num = int(f.readline().rstrip())
  valid_face_num = 0

  for i in range(face_num):
    annot = f.readline().rstrip().split()
    # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
    if(float(annot[2]) > 25.0):
        if(float(annot[3]) > 30.0):
            xmins.append( max(0.005, (float(annot[0]) / width) ) )
            ymins.append( max(0.005, (float(annot[1]) / height) ) )
            xmaxs.append( min(0.995, ((float(annot[0]) + float(annot[2])) / width) ) )
            ymaxs.append( min(0.995, ((float(annot[1]) + float(annot[3])) / height) ) )
            classes_text.append('face')
            classes.append(1)
            poses.append("front".encode('utf8'))
            truncated.append(int(0))
            print(xmins[-1], ymins[-1], xmaxs[-1], ymaxs[-1], classes_text[-1], classes[-1])
            valid_face_num += 1;

  print("Face Number is %d" % face_num)
  print("Valid face number is %d" % valid_face_num)


  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(int(height)),
    'image/width': dataset_util.int64_feature(int(width)),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/difficult': dataset_util.int64_list_feature(int(0)),
    'image/object/truncated': dataset_util.int64_list_feature(truncated),
    'image/object/view': dataset_util.bytes_list_feature(poses),
    }))


  return valid_face_num, tf_example


def main(unused_argv):
  f = open("WIDER/wider_face_train_annot.txt")
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

# WIDER FACE DATASET ANNOTATED 12880 IMAGES
  valid_image_num = 0
  invalid_image_num = 0
  for image_idx in range(12880):
    print("image idx is %d" % image_idx)
    valid_face_number, tf_example = parse_example(f)
    if(valid_face_number != 0):
      writer.write(tf_example.SerializeToString())
      valid_image_num += 1
    else:
      invalid_image_num += 1
      print("Pass!")
  writer.close()

  print("Valid image number is %d" % valid_image_num)
  print("Invalid image number is %d" % invalid_image_num)



if __name__ == '__main__':
  tf.app.run()
