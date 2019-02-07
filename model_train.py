import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil

from layers import LayerTrain

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



	#####training#########3


if __name__ == "__main__":
	main()