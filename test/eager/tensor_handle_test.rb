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

      def test_add
        x = TensorHandle.new(Tensor.new(7))
        y = x + 3
        assert_equal(10, y.value)
      end

      def test_subtract
        x = TensorHandle.new(Tensor.new(7))
        y = x - 3
        assert_equal(4, y.value)
      end

      def test_multiply
        x = TensorHandle.new(Tensor.new(7))
        y = x * 3
        assert_equal(21, y.value)
      end

      def test_divide
        x = TensorHandle.new(Tensor.new(9))
        y = x / 3
        assert_equal(3, y.value)
      end

      def test_negative
        x = TensorHandle.new(Tensor.new(9))
        y = -x
        assert_equal(-9, y.value)
      end

      def test_exponent
        x = TensorHandle.new(Tensor.new(9))
        y = x ** 3
        assert_equal(729, y.value)
      end

      def test_modulus
        x = TensorHandle.new(Tensor.new(9))
        y = x % 7
        assert_equal(2, y.value)
      end
    end
  end
end