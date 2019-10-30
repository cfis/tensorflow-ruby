require_relative "../test_helper"

module Tensorflow
  class LinalgTest < Minitest::Test
    def test_matmul_eager
      x = [[2.0]]
      assert_equal([[4.0]], Tf.matmul(x, x).value)
      assert_equal([[4.0]], Tf::Linalg.matmul(x, x).value)
    end

    def test_matmul_graph
      x = Graph::Graph.default.constant([[2.0]])
      matmul = Tf.matmul(x, x)
      result = Graph::Graph.default.execute(outputs: matmul)
      assert_equal([[4.0]], result)
    end
  end
end