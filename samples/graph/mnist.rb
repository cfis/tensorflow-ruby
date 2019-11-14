$LOAD_PATH.unshift(File.expand_path('../../lib', __dir__))

require "tensorflow"
require_relative "../../lib/datasets/images/mnist"

mnist = Tensorflow::Datasets::Images::Mnist.new

# Mnist images are 28 by 28 pixels
n_inputs = 28*28


n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
mnist.train_dataset


Tf.disable_eager_execution



