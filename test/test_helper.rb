require "bundler/setup"
#Bundler.require(:default)
require "minitest/autorun"
require "tensorflow"
require "tensorflow/extensions/array"

class MyModel < Tensorflow::Keras::Model
  def initialize
    super
    @flatten = Tensorflow::Keras::Layers::Flatten.new(input_shape: [28, 28])
    @d1 = Tensorflow::Keras::Layers::Dense.new(128, activation: "relu")
    @dropout = Tensorflow::Keras::Layers::Dropout.new(0.2)
    @d2 = Tensorflow::Keras::Layers::Dense.new(10, activation: "softmax")
  end

  def call(x)
    x = @flatten.call(x)
    x = @d1.call(x)
    x = @dropout.call(x)
    @d2.call(x)
  end
end
