require_relative "../test_helper"

module Tensorflow
  class LinalgTest < Minitest::Test
    def test_matmul_eager
      Tensorflow.enable_eager_execution
      x = [[2.0]]
      assert_equal([[4.0]], Tensorflow.matmul(x, x).value)
      assert_equal([[4.0]], Linalg.matmul(x, x).value)
    end

    def test_matmul_graph
      Tensorflow.disable_eager_execution
      x = Tensorflow.constant([[2.0]])
      matmul = Tensorflow.matmul(x, x)
      result = Graph::Graph.default.execute([matmul])
      assert_equal([[4.0]], result)
    end
  end
end