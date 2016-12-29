#!/usr/bin/env sh
#
# This script downloads VGG Face Descriptor[http://www.robots.ox.ac.uk/~vgg/software/vgg_face/]

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $DIR
cd $DIR

echo "Downloading.."
wget -c http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz

echo "Unzipping.."
tar -xf vgg_face_caffe.tar.gz && rm -f vgg_face_caffe.tar.gz
