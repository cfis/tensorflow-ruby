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

# Ops
require "tensorflow/ops/audio"
require "tensorflow/ops/bitwise"
require "tensorflow/ops/control"
require "tensorflow/ops/image"
require "tensorflow/ops/io"
require "tensorflow/ops/linalg"
require "tensorflow/ops/math"
require "tensorflow/ops/nn"
require "tensorflow/ops/operators"
require "tensorflow/ops/ops"
require "tensorflow/ops/random"
require "tensorflow/ops/raw_ops"

# Core
require "tensorflow/ffi"
require "tensorflow/decorators"
require "tensorflow/error"
require "tensorflow/execution_context"
require "tensorflow/name_scope"
require "tensorflow/op_def_builder"
require "tensorflow/status"
require "tensorflow/strings"
require "tensorflow/tensor_mixin"
require "tensorflow/tensor_data"
require "tensorflow/tensor"
require "tensorflow/variable"
require "tensorflow/version"

# Extensions
require "tensorflow/extensions/arg_def.rb"
require "tensorflow/extensions/boolean.rb"
require "tensorflow/extensions/narray.rb"

# Printers
require "tensorflow/printers/graph"

# Eager
require "tensorflow/eager/context"
require "tensorflow/eager/operation"
require "tensorflow/eager/tensor_handle"

# graph
require "tensorflow/graph/function"
require "tensorflow/graph/function_def"
require "tensorflow/graph/gradients"
require "tensorflow/graph/graph"
require "tensorflow/graph/graph_def_options"
require "tensorflow/graph/graph_keys"
require "tensorflow/graph/operation"
require "tensorflow/graph/operation_output"
require "tensorflow/graph/operation_attr"
require "tensorflow/graph/operation_description"
require "tensorflow/graph/session"

# Ugly - now require the op gradients
require "tensorflow/ops/gradients"


# specs
require "tensorflow/type_spec"
require "tensorflow/batchable_type_spec"
require "tensorflow/tensor_spec"

# Train
require "tensorflow/train/optimizer"
require "tensorflow/train/gradient_descent_optimizer"

# data
require "tensorflow/data/dataset"
require "tensorflow/data/batch_dataset"
require "tensorflow/data/fixed_length_record_dataset"
require "tensorflow/data/iterator"
require "tensorflow/data/map_dataset"
require "tensorflow/data/repeat_dataset"
require "tensorflow/data/shuffle_dataset"
require "tensorflow/data/tensor_dataset"
require "tensorflow/data/tensor_slice_dataset"
require "tensorflow/data/zip_dataset"

# keras
require "tensorflow/keras/utils"
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

require 'tensorflow/python_compatiblity'

module Tensorflow
  extend Ops

  GRAPH_MODE = 0
  EAGER_MODE = 1

  class << self
    attr_accessor :ffi_lib

    def library_version
      FFI.TF_Version
    end
  end

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

  def self.execution_mode
    @mode ||= Tensorflow::EAGER_MODE
  end

  def self.execution_mode=(value)
    @mode = value
  end

  def self.name_scope(base_name, &block)
    ExecutionContext.current.name_scope(base_name, &block)
  end

  extend PythonCompatability
  class << self
    extend Forwardable
    def_delegators Linalg, *Linalg.singleton_methods
    def_delegators Math, *Math.singleton_methods
    def_delegators NN, *NN.singleton_methods
  end
end