import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil

class TrainLayer(object):
	

	def __init__(self, in_depth, out_depth, N_clusters, name)
		self.name = name

		if 'conv' in name:
			self.w = tf.Variable(tf.random_normal([5, 5, in_depth, out_depth], stddev = 0.1))

		if 'fc' in name:
			self.w = tf.Variable(tf.random_normal([in_depth, out_depth], stddev = 0.1))

		self.place = tf.placeholder(tf.float32, self.w.shape)
		self.assign = tf.assign(self.w, self.place)
		self.num_weights = np.prod(self.w.shape)

		#### for pruning masking the placeholder
		#### 1 --> Valid weights
		#### 2 --> Invalid weights

		self.mask = np.ones(self.w.shape, dtype=np.float32)

		###### clusters are usedd fro quantization
		self.N_clusters = N_clusters

		def forward(self, x):