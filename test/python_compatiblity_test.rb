require_relative "test_helper"

module Tensorflow
  class PythonCompatiblityTest < Minitest::Test
    def test_tf
      assert_same(Tensorflow, Tf)
    end

    def test_set_mode
      Tensorflow.disable_eager_execution
      assert_equal(Tensorflow::GRAPH_MODE, Tensorflow.execution_mode)

      Tensorflow.enable_eager_execution
      assert_equal(Tensorflow::EAGER_MODE, Tensorflow.execution_mode)
    end
  end
end