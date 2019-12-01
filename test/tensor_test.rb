require_relative "base_test"

module Tensorflow
  class TensorTest < BaseTest
    def setup
      Tensorflow.execution_mode = Tensorflow::EAGER_MODE
    end

    def test_float
      value = 1.0
      tensor = Tensor.new(value)
      assert_equal(:float, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_float_array
      value = [2.5, 3.5]
      tensor = Tensor.new(value, dtype: :float)
      assert_equal(:float, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_shape
      tensor = Tensor.new(1, shape: [5, 30])
      assert_equal([5, 30], tensor.shape)
      assert_equal(:int32, tensor.dtype)
      assert_equal(1, tensor.value[4,29])
    end

    def test_numo_1d
      narray = Numo::NArray[1, 2, 3, 4]
      tensor = Tensor.new(narray)
      assert_equal([4], tensor.shape)
      assert_equal(:int32, tensor.dtype)
      assert_equal([1, 2, 3, 4], tensor.value)
    end

    def test_narray_2d
      narray = Numo::NArray[[1], [2], [3], [4]]
      tensor = Tensor.new(narray)
      assert_equal([4, 1], tensor.shape)
      assert_equal(:int32, tensor.dtype)
      assert_equal([[1], [2], [3], [4]], tensor.value)
    end

    def test_add
      x = Tensor.new(7)
      y = x + 3
      assert_equal(10, y.value)
    end

    def test_subtract
      x = Tensor.new(7)
      y = x - 3
      assert_equal(4, y.value)
    end

    def test_multiply
      x = Tensor.new(7)
      y = x * 3
      assert_equal(21, y.value)
    end

    def test_divide
      x = Tensor.new(9)
      y = x / 3
      assert_equal(3, y.value)
    end

    def test_negative
      x = Tensor.new(9)
      y = -x
      assert_equal(-9, y.value)
    end

    def test_exponent
      x = Tensor.new(9)
      y = x ** 3
      assert_equal(729, y.value)
    end

    def test_modulus
      x = Tensor.new(9)
      y = x % 7
      assert_equal(2, y.value)
    end

    def test_proto
      data = "\b\x03\x12\x00\"\x04*\x00\x00\x00"
      proto = TensorProto.decode(data)
      tensor = Tensor.from_proto(proto)
      assert_equal([], tensor.shape)
      assert_equal(42, tensor.value)
    end
  end
end 
