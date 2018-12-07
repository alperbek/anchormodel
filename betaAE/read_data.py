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

	def __init__(self,batch_size=32, repeat = True, shuffle=True, shuffle_size=800):
		self.training_filenames = ["/e/DATA/Matterport/Matteport_BBOX_TFRECORD_7CLASS/matterport_bbox_300_train_{0:05d}-of-00100.tfrecord".format(i) for i in range(100)]
		self.validation_filenames = ["/e/DATA/Matterport/Matteport_BBOX_TFRECORD_7CLASS/matterport_bbox_300_validation_{0:05d}-of-00100.tfrecord".format(i) for i in range(100)]

		#self.filenames = tf.placeholder(tf.string,shape=[None])
		self.dataset = tf.data.TFRecordDataset(self.training_filenames)
		self.dataset = self.dataset.map(self._parse)
		if(shuffle):
			self.dataset = self.dataset.shuffle(shuffle_size)
		self.dataset = self.dataset.batch(batch_size,drop_remainder=True)
		if(repeat):
			self.dataset = self.dataset.repeat()
		#self.iterator = self.dataset.make_initializable_iterator()

	def _parse(self,example_proto):
			# Defaults are not specified since both keys are required.
		features={
			'image/height': tf.FixedLenFeature([], tf.int64),
			'image/format':tf.FixedLenFeature([],tf.string),
			'image/width': tf.FixedLenFeature([], tf.int64),
			'image/encoded': tf.FixedLenFeature([], tf.string),
			'image/bbox': tf.VarLenFeature(tf.float32),
			'image/classes'	:tf.VarLenFeature(tf.int64),
			'image/num_instance':tf.FixedLenFeature([],tf.int64)
			}
		parsed_features = tf.parse_single_example(example_proto,features)
		
		bboxes = parsed_features['image/bbox']
		classes = parsed_features['image/classes']
		
		#bboxes = tf.sparse_tensor_to_dense(parsed_features['image/bbox'], default_value=0)
		#classes = tf.sparse_tensor_to_dense(parsed_features['image/classes'],default_value=0)
		num_instance = tf.cast(parsed_features["image/num_instance"],tf.int32)
		classes = tf.cast(classes,tf.int32)
		#bboxes = tf.reshape(bboxes,[num_instance,5])
		image = tf.cast(tf.image.decode_image(parsed_features["image/encoded"],channels=3),tf.float32)
		height = tf.cast(parsed_features["image/height"],tf.int32)
		width = tf.cast(parsed_features["image/width"],tf.int32)
		image_shape = tf.stack([300, 300,3])
		image = tf.reshape(image,image_shape)
		image = tf.image.resize_images(image,[256,256])
		image = image /255.0 -0.5
		return image, image  # Return the image itself as the target for BVAE. // CAGRI October 30th 2018


	def getDataIterator(self):

		return self.iterator


	def nextBatch(self):
		images, labels = self.iterator.get_next()
		return images, labels


class VRDataSet_AE(object):
	DEPTH = 0
	RGB = 1
	NORMAL = 2

	def __init__(self, batch_size=32, shuffle=True, repeat=True, shuffle_size=800, modality=DEPTH):
	
		if modality == self.DEPTH:
			self.filenames = tf.constant(["/d/Mediate/AI/DataPreprocessing/TestData/depth/depth{}.png".format(i) for i in range(5999)])	
		elif modality == self.RGB:
			self.filenames = tf.constant(["/d/Mediate/AI/DataPreprocessing/TestData/lit/lit{}.png".format(i) for i in range(5999)])	
		elif modality == self.NORMAL:
			self.filenames = tf.constant(["/d/Mediate/AI/DataPreprocessing/TestData/normal/normal{}.png".format(i) for i in range(5999)])	

		self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames))
		self.dataset = self.dataset.map(self._parse_function)
		if(shuffle):
			self.dataset = self.dataset.shuffle(shuffle_size)
		self.dataset = self.dataset.batch(batch_size,drop_remainder=True)
		if(repeat):
			self.dataset = self.dataset.repeat()
		#self.depth_filenames =["/e/DATA/Matterport/Matteport_BBOX_TFRECORD_7CLASS/matterport_bbox_300_train_{0:05d}-of-00100.tfrecord".format(i) for i in range(100)]

	def _parse_function(self, filename):
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_png(image_string, channels=3)
		image_resized = tf.image.resize_images(image_decoded, [256, 256])
		image_normalized = image_resized /255.0 -0.5
		return image_normalized, image_normalized

class ToyDatasetShapes_AE(object):
	DEPTH = 0
	RGB = 1
	NORMAL = 2

	def __init__(self, batch_size=32, output_size=28,shuffle=True, repeat=False, shuffle_size=800, modality=DEPTH):

		self.filenames = tf.constant(["/d/PhD/AnchorModel/ToyDataShapes/sketch_181113a/{}.jpg".format(i) for i in range(3000)])	
		self.test_filenames = tf.constant(["/d/PhD/AnchorModel/ToyDataShapes/sketch_181113a/{}.jpg".format(i+3000) for i in range(2000)])
		self.output_size=output_size
		self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames))
		self.dataset = self.dataset.map(self._parse_function)
		
		self.test_dataset = tf.data.Dataset.from_tensor_slices((self.test_filenames))
		self.test_dataset = self.test_dataset.map(self._parse_function)

		if(shuffle):
			self.dataset = self.dataset.shuffle(shuffle_size)
		self.dataset = self.dataset.batch(batch_size,drop_remainder=True)
		self.test_dataset=self.test_dataset.batch(batch_size,drop_remainder=True)
		if(repeat):
			self.dataset = self.dataset.repeat()
		#self.depth_filenames =["/e/DATA/Matterport/Matteport_BBOX_TFRECORD_7CLASS/matterport_bbox_300_train_{0:05d}-of-00100.tfrecord".format(i) for i in range(100)]


	def _parse_function(self, filename):
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_png(image_string, channels=3)
		image_resized = tf.image.resize_images(image_decoded, [self.output_size, self.output_size])
		image_normalized = image_resized /255.0
		return image_normalized, image_normalized
