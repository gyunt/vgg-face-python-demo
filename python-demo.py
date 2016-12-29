from __future__ import division
from __future__ import print_function

import caffe
import cv2
import numpy as np

names = []


def read_name_list():
    with open("./vgg_face_caffe/names.txt") as f:
        global names
        names = f.readlines()


def main():
    read_name_list()

    image = np.array(cv2.imread("./vgg_face_caffe/ak.png")).astype(np.float32)
    averageImg = [129.1863, 104.7624, 93.5940]

    image[:][:][0] -= averageImg[0]
    image[:][:][1] -= averageImg[1]
    image[:][:][2] -= averageImg[2]

    model = './vgg_face_caffe/VGG_FACE_deploy.prototxt'
    weights = './vgg_face_caffe/VGG_FACE.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(model, weights, caffe.TEST)

    net.blobs['data'].data[...] = image.transpose(2, 1, 0).reshape([1, 3, 224, 224])
    net.forward()

    prob = net.blobs['prob'].data[0]
    caffe_ft = net.blobs['fc8'].data[0]

    max_prob = -1.0
    max_index = -1
    for index in range(len(prob)):
        if max_prob < caffe_ft[index]:
            max_prob = caffe_ft[index]
            max_index = index

    global names
    print("name: ", names[max_index+1], "index: ", max_index)

if __name__ == '__main__':
    main()

