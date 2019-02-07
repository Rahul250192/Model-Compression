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
		if 'conv' in self.name:
			return tf.nn.conv2d(x, self.w, strides=[1,2,2,1], padding='SAME')
		elif 'fc' in self.name:
			return tf.matmul(x, self.w)

	def save_histogram(self, sess, directory, it):
		w_data = sess.run(self.w).reshape(-1)
		valid_w = [x for x in w_data if x!=0.0]

		plt.grid(True)
		plt.hist(valid_w, 100, color='0.4')
		plt.gca().set_xlin([-0.3, 0.3])
		plt.savfig(directory + '/' + self.name + '-' + str(it), dpi=100)
		plt.gcf().clear

	def save_weights(self, sess, dire):
		w_data = sess.run(self.w)
		np.save(dire + '/' + self.name + '-weights', w_data)
		np.save(dire + '/' + self.name + '-prune--Weights', self.mask)

	def pruning(self, sess, th):
		w_data = sess.run(self.w)
		self.mask = (np.abs(w_data) > = th).astype(np.float32)

		print('layer:', self.name)
		print('\tremaining weights', int(np.sum(self.mask)))
		print('total weights', self.num_weights)

		sess.run(self.assign, feed_dict=[self.place: self.mask*w_data])

	def prune_weights(self, grad):
		return grad * self.mask

	def weight_update(self, sess):
		w_data = sess.run(self.w)
		sess.run(self.assign, feed_dict=[self.place: self.mask*w_data])

	def quantize(self, sess):
		w_data = sess.run(self.w)

		max_v, min_v = np.max(w_data), np.min(w_data)
		self.centroids = np.linspace(min_v, max_v, self.N_clusters)
		w_data = np.expand_dims(w_data, 0)

		centroids_old = np.copy(self.centroids)

		for i in range(20):

			if 'conv' in self.name:
				dis = np.abs(w_data - np.reshape(self.centroids, (-1,1,1,1,1)))
				dis = np.transpose(dis, (1,2,3,4,0))

			elif 'fc' in self.name:
				dis = np.abs(w_data - np.reshape(self.centroids, (-1,1,1)))
				dis = np.transpose(dis, (1,2,0))

			classes = np.argmin(dis, axis=-1)

			self.cluster_mask = []
			for i in range(self.N_clusters):
				cluster_mask 