# dependencies
require "ffi"
require "npy"
require "numo/narray"

# stdlib
require "digest"
require "fileutils"
require "forwardable"
require "json"
require "net/http"
require "tempfile"
require "zlib"

# Load protobufs. Is this a good idea to require all ruby files in those directories?
Dir[File.join(__dir__, 'tensorflow', 'core', 'lib', 'core', '*.rb')].each { |file| require file }
Dir[File.join(__dir__, 'tensorflow', 'core', 'stream_executor', '*.rb')].each { |file| require file }
Dir[File.join(__dir__, 'tensorflow', 'core', 'framework', '*.rb')].each { |file| require file }
#Dir[File.join(__dir__, 'tensorflow', 'core', 'protobuf', '*.rb')].each { |file| require file }

# Core
require "tensorflow/op_def_builder"
require "tensorflow/status"
require "tensorflow/strings"
require "tensorflow/tensor_data_pointer"
require "tensorflow/tensor_mixin"
require "tensorflow/tensor"
require "tensorflow/utils"
require "tensorflow/variable"
require "tensorflow/version"

# Ops
require "tensorflow/ops/audio"
require "tensorflow/ops/bitwise"
require "tensorflow/ops/image"
require "tensorflow/ops/io"
require "tensorflow/ops/linalg"
require "tensorflow/ops/math"
require "tensorflow/ops/nn"
require "tensorflow/ops/ops"
require "tensorflow/ops/random"
require "tensorflow/ops/raw_ops"

# Printers
require "tensorflow/printers/graph"

# Eager
require "tensorflow/eager/eager"
require "tensorflow/eager/context"
require "tensorflow/eager/operation"
require "tensorflow/eager/tensor_handle"

# graph
require "tensorflow/graph/function"
require "tensorflow/graph/gradients"
require "tensorflow/graph/graph"
require "tensorflow/graph/graph_def_options"
require "tensorflow/graph/name_scope"
require "tensorflow/graph/operation"
require "tensorflow/graph/operation_description"
require "tensorflow/graph/session"

# specs
require "tensorflow/type_spec"
require "tensorflow/batchable_type_spec"
require "tensorflow/tensor_spec"

# data
require "tensorflow/data/dataset"
require "tensorflow/data/batch_dataset"
require "tensorflow/data/fixed_length_record_dataset"
require "tensorflow/data/map_dataset"
require "tensorflow/data/repeat_dataset"
require "tensorflow/data/shuffle_dataset"
require "tensorflow/data/tensor_dataset"
require "tensorflow/data/tensor_slice_dataset"
require "tensorflow/data/zip_dataset"

# keras
require "tensorflow/keras/datasets/boston_housing"
require "tensorflow/keras/datasets/cifar10"
require "tensorflow/keras/datasets/cifar100"
require "tensorflow/keras/datasets/fashion_mnist"
require "tensorflow/keras/datasets/imdb"
require "tensorflow/keras/datasets/mnist"
require "tensorflow/keras/datasets/reuters"
require "tensorflow/keras/layers/conv"
require "tensorflow/keras/layers/conv2d"
require "tensorflow/keras/layers/dense"
require "tensorflow/keras/layers/dropout"
require "tensorflow/keras/layers/flatten"
require "tensorflow/keras/losses/sparse_categorical_crossentropy"
require "tensorflow/keras/metrics/mean"
require "tensorflow/keras/metrics/sparse_categorical_accuracy"
require "tensorflow/keras/model"
require "tensorflow/keras/models/sequential"
require "tensorflow/keras/optimizers/adam"
require "tensorflow/keras/preprocessing/image"
require "tensorflow/keras/utils"

require 'tensorflow/core/framework/op_def_pb'

# We can't use Tensorflow::Error because a protobuf message annoyingly assigns that as a module
class TensorflowError < StandardError
end

module Tensorflow
  class << self
    attr_accessor :ffi_lib
  end
  self.ffi_lib = ["tensorflow", "libtensorflow.so"]

  # friendlier error message
  autoload :FFI, "tensorflow/ffi"

  def self.op_defs
    buffer = FFI.TF_GetAllOpList
    string = buffer[:data].read_string(buffer[:length])
    ops = OpList.decode(string)
    ops.op.each_with_object(Hash.new) do |op_def, hash|
      hash[op_def.name] = op_def
    end
  ensure
    FFI.TF_DeleteBuffer(buffer)
  end

  def self.op_def(op_name)
    self.op_defs[op_name]
  end

  class << self
    include Ops
    include Utils

    extend Forwardable
    def_delegators Linalg, :eye, :matmul
    def_delegators Math, :abs, :acos, :acosh, :add, :add_n, :argmax, :argmin, :asin, :asinh, :atan, :atan2, :atanh, :cos, :cosh, :cumsum, :divide, :equal, :exp, :floor, :greater, :greater_equal, :less, :less_equal, :logical_and, :logical_not, :logical_or, :maximum, :minimum, :multiply, :negative, :not_equal, :pow, :reduce_all, :reduce_any, :reduce_logsumexp, :reduce_max, :reduce_mean, :reduce_min, :reduce_prod, :reduce_sum, :round, :scalar_mul, :sigmoid, :sign, :sin, :sinh, :sqrt, :square, :subtract, :tan, :tanh, :truediv
    def_delegators NN, :space_to_batch
    def_delegators Tensor, :constant, :placeholder

    def library_version
      FFI.TF_Version
    end
  end
end

# shortcut
Tf = Tensorflow
