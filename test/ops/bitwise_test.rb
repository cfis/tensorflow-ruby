require_relative "../base_test"

module Tensorflow
  class BitwiseTest < BaseTest
    def setup
      Tensorflow.execution_mode = Tensorflow::EAGER_MODE
    end

    def test_bitwise_and_eager
      lhs = Tensor.new([0, 5, 3, 14])
      rhs = Tensor.new([5, 0, 7, 11])
      result = Bitwise.bitwise_and(lhs, rhs)
      assert_equal([0, 0, 3, 10], result.value)
    end

    def test_bitwise_and_graph
      Graph::Graph.new.as_default do |graph|
        lhs = Tensorflow.constant([0, 5, 3, 14])
        rhs = Tensorflow.constant([5, 0, 7, 11])
        operation = Bitwise.bitwise_and(lhs, rhs)
        result = graph.execute(operation)
        assert_equal([0, 0, 3, 10], result)
      end
    end
  end
end