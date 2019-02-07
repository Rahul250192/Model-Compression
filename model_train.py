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

	###################Import Data###############################33

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

	#####training#########
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	itr = []
	itr_acc = []

	for i in range(600):

		batch_x, batch_y = mnist.train.next_batch(50)
		batch_x = np.reshape(batch_x,(-1, 28, 28, 1))

		feed_dict = {x_placeholder: batch_x, labels: batch_y}


		if i < 200:
			sess.run(train_step, feed_dict=feed_dict)

		elif i>=200 and i < 400:

			if i%500 = 0:
				print(i, 'pruning')
				for l in Layers:
					l.pruning(sess, 0.1) # thresold

			g_data = sess.run(grads, feed_dict={x_placeholder:batch_x, labels:batch_y})
			feed_dict = {}
			for l , g, gs in zip(Layers, grads, g_data):
				pruned_grad = l.prune_weights_grad(gs)
				feed_dict[g] = pruned_grad

			sess.run(train_step, feed_dict=feed_dict)

			for l in Layers:
				l.weight_update(sess)

	#################quantize##############################

		else:

			if i ==1000:
				print(i, "quantize w")
				for l in Layers:
					l.quantize(sess)

			g_data = sess.run(grads, feed_dict={x_placeholder:batch_x, labels:batch_y})
			feed_dict = {}
			for l , g, gs in zip(Layers, grads, g_data):
				quantized_grad = l.quantized_grad(gs)
				feed_dict[g] = quantized_grad


			sess.run(train_step, feed_dict=feed_dict)

			for l in Layers:
				l.quantize_cen_update(sess)
				l.quantize_w_update(sess)


	


if __name__ == "__main__":
	main()