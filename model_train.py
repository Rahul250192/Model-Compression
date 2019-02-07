import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil

from layers import TrainLayer

def make_dir(dire):

	if os.path.exists(dire):
		shutil.rmtree(dire, ignore_errors=True)
	os.makedirs(dire)

def main():
	hist_dir = './histograms'
	weight_dir = './weights'

	make_dir(hist_dir)
	make_dir(weight_dir)

	#####creating computation graph######
	L1 = TrainLayer(1, 32, N_clusters=5, name='convL1')
	L2 = TrainLayer(32, 64, N_clusters=5, name='convL2')
	L3 = TrainLayer(7 * 7 * 64, 1024, N_clusters=5, name='fcL1')
	L4 = TrainLayer(1024, 10, N_clusters=5, name='fcL2')

	Layers = [L1, L2, L3, L4]
	Layers_weights = [L1.w, L2.w, L3.w, L4.w]

	x_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
	x = tf.nn.relu(L1.forward(x_placeholder))
	x = tf.nn.relu(L2.forward(x))
	x = tf.reshape(x, (-1, int(np.product(x.shape[1:]))))
	x = tf.nn.relu(L3.forward(x))
	logits = L4.forward(x)


	preds = tf.nn.softmax(logits)
	labels = tf.placeholder(tf.float32, [None,10])
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

	optimizer = tf.train.AdamOptimizer(1e-4)
	gradiant_var = optimizer.compute_gradients(loss, Layers_weights)
	grads = [grad for grad, var in gradiant_var]
	train_step = optimizer.apply_gradients(gradients_var)


	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	#####training#########


if __name__ == "__main__":
	main()