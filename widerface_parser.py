import os
import numpy
import cv2

f = open("WIDER/wider_face_train_annot.txt")
# Total annotated image number is 12880
for j in range(12880):
  filename = f.readline().rstrip()
  print(filename)

  filepath = os.path.join("./WIDER/WIDER_train/images/", filename)
  print(filepath)

  encoded = open(filepath).read()
  image = cv2.imread(filepath)
  height, width, channel = image.shape
  print("height is %d, width is %d, channel is %d" % (height, width, channel))

  face_num = int(f.readline().rstrip())
  print(face_num)

  tlx = []
  tly = []
  w = []
  h = []

  for i in range(face_num):
    annot = f.readline().rstrip().split()

    tlx.append(float(annot[0]))
    tly.append(float(annot[1]))
    w.append(float(annot[2]))
    h.append(float(annot[3]))

    print(tlx[i], tly[i], w[i], h[i])

  print(j)
