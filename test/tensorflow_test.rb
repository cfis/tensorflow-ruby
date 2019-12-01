require_relative "base_test"

module Tensorflow
  class TensorflowTest < BaseTest
    def test_version
      assert_equal "2.0.0", Tensorflow.library_version
    end

    def test_operations
      op_defs = Tensorflow.op_defs
      assert_kind_of(Hash, op_defs)
      assert(op_defs.keys.length > 1000)
    end

    def test_op_def
      op_def = Tensorflow.op_def('ZipDataset')
      refute_nil(op_def)
    end

    def test_set_mode
      Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
      assert_equal(Tensorflow::GRAPH_MODE, Tensorflow.execution_mode)

      Tensorflow.execution_mode = Tensorflow::EAGER_MODE
      assert_equal(Tensorflow::EAGER_MODE, Tensorflow.execution_mode)
    end
  end
end