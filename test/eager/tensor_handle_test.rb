require_relative "../test_helper"

module Tensorflow
  module Eager
    class TensorHandleTest < Minitest::Test
      def test_tensor
        tensor = Tensor.new("Some tensor")

        handle = TensorHandle.new(tensor)
        assert_equal(tensor.value, handle.tensor.value)
      end

      def test_dtype
        tensor = Tensor.new(false)
        handle = TensorHandle.new(tensor)
        assert_equal(:bool, handle.dtype)
      end

      def test_dims
        tensor = Tensor.new([[1,2],
                             [3,4]])

        handle = TensorHandle.new(tensor)
        assert_equal([2, 2], handle.shape)
      end

      def test_element_count
        tensor = Tensor.new([[1,2],
                             [3,4]])

        handle = TensorHandle.new(tensor)
        assert_equal(4, handle.element_count)
      end
    end
  end
end

