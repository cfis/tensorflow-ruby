require_relative "../test_helper"

module Tensorflow
  class BitwiseTest < Minitest::Test
    def test_bitwise_and
      lhs = Tensor.new([0, 5, 3, 14])
      rhs = Tensor.new([5, 0, 7, 11])
      assert_equal [0, 0, 3, 10], Bitwise.bitwise_and(lhs, rhs).value
    end
  end
end