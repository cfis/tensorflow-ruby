require_relative "test_helper"

module Tensorflow
  class TensorTest < Minitest::Test
    def test_float
      value = 1.0
      tensor = Tensor.new(value)
      assert_equal(:float, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_true
      value = true
      tensor = Tensor.new(value)
      assert_equal(:bool, tensor.dtype)
      assert(tensor.value)
    end

    def test_false
      value = false
      tensor = Tensor.new(value)
      assert_equal(:bool, tensor.dtype)
      refute(tensor.value)
    end

    def test_boolean_array
      value = [[true, false], [false, true]]
      tensor = Tensor.new(value)
      assert_equal(:bool, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_int32
      value = 1
      tensor = Tensor.new(value)
      assert_equal(:int32, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_int32_2
      value = 2147483647
      tensor = Tensor.new(value)
      assert_equal(:int32, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_int32_3
      value = 2147483647
      tensor = Tensor.new(value)
      assert_equal(:int32, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_int32_array
      value = [[1,2,3], [3,4,5]]
      tensor = Tensor.new(value)
      assert_equal([2, 3], tensor.shape)
      assert_equal(:int32, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_int64
      value = 2147483648
      tensor = Tensor.new(value)
      assert_equal(:int64, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_int64_negative
      value = -2147483649
      tensor = Tensor.new(value)
      assert_equal(:int64, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_complex_128
      value = Complex(2, 3)
      tensor = Tensor.new(value)
      assert_equal(:complex128, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_string
      value = "hello"
      tensor = Tensor.new(value)
      assert_equal(:string, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_string_array
      value = ["Hi", "there"]
      tensor = Tensor.new(value)
      assert_equal(:string, tensor.dtype)
      assert_equal(value, tensor.value)
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
  end
end 
