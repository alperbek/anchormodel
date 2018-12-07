from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import argparse
import json
import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class MatterportDataset3DBBOX(object):

	def __init__(self, batch_size=32, repeat=True, shuffle=False, shuffle_size=800):
		self.training_filenames = ["/e/DATA/Matterport/Matteport_BBOX_TFRECORD_7CLASS/matterport_bbox_300_train_{0:05d}-of-00100.tfrecord".format(i) for i in range(100)]
		self.validation_filenames = ["/e/DATA/Matterport/Matteport_BBOX_TFRECORD_7CLASS/matterport_bbox_300_validation_{0:05d}-of-00100.tfrecord".format(i) for i in range(100)]

		self.filenames = tf.placeholder(tf.string, shape=[None])
		dataset = tf.data.TFRecordDataset(self.filenames)
		dataset = dataset.map(self._parse)
		if(shuffle):
			dataset = dataset.shuffle(shuffle_size)
		dataset = dataset.batch(batch_size, drop_remainder=True)
		if(repeat):
			dataset = dataset.repeat()
		self.iterator = dataset.make_initializable_iterator()

	def _parse(self,example_proto):
			# Defaults are not specified since both keys are required.
		features ={
			'image/height': tf.FixedLenFeature([], tf.int64),
			'image/format': tf.FixedLenFeature([], tf.string),
			'image/width': tf.FixedLenFeature([], tf.int64),
			'image/encoded': tf.FixedLenFeature([], tf.string),
			'image/bbox': tf.VarLenFeature(tf.float32),
			'image/classes'	: tf.VarLenFeature(tf.int64),
			'image/num_instance': tf.FixedLenFeature([], tf.int64)
			}
		parsed_features = tf.parse_single_example(example_proto, features)
		
		bboxes = parsed_features['image/bbox']
		classes = parsed_features['image/classes']
		
		#bboxes = tf.sparse_tensor_to_dense(parsed_features['image/bbox'], default_value=0)
		#classes = tf.sparse_tensor_to_dense(parsed_features['image/classes'],default_value=0)
		num_instance = tf.cast(parsed_features["image/num_instance"],tf.int32)
		classes = tf.cast(classes,tf.int32)
		#bboxes = tf.reshape(bboxes,[num_instance,5])
		image = tf.cast(tf.image.decode_image(parsed_features["image/encoded"],channels=3),tf.float32)
		image=tf.image.rgb_to_grayscale(image)
		height = tf.cast(parsed_features["image/height"],tf.int32)
		width = tf.cast(parsed_features["image/width"],tf.int32)
		image_shape = tf.stack([300, 300,1])
		image = tf.reshape(image,image_shape)
		image = tf.image.resize_images(image,[28,28])
		return image, bboxes, classes, num_instance


	def getDataIterator(self):

		return self.iterator


	def nextBatch(self):
		images, bboxes,classes,num_instance = self.iterator.get_next()
		bboxes = tf.sparse_tensor_to_dense(bboxes, default_value=-1)
		classes = tf.sparse_tensor_to_dense(classes,default_value=-1)
		return images,bboxes,classes,num_instance


