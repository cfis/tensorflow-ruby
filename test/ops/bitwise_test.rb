require_relative "../test_helper"

module Tensorflow
  class BitwiseTest < Minitest::Test
    def test_bitwise_and_eager
      lhs = Tensor.new([0, 5, 3, 14])
      rhs = Tensor.new([5, 0, 7, 11])
      result = Bitwise.bitwise_and(lhs, rhs)
      assert_equal([0, 0, 3, 10], result.value)
    end

    def test_bitwise_and_graph
      lhs = Graph::Graph.default.constant([0, 5, 3, 14])
      rhs = Graph::Graph.default.constant([5, 0, 7, 11])
      operation = Bitwise.bitwise_and(lhs, rhs)
      result = Graph::Graph.default.execute(outputs: operation)
      assert_equal([0, 0, 3, 10], result)
    end
  end
end