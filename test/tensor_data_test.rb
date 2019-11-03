require_relative "test_helper"

module Tensorflow
  class TensorDataTest < Minitest::Test
    def test_figure_int32_dtype
      result = TensorData.figure_dtype(32)
      assert_equal(:int32, result)
    end

    def test_figure_int64_dtype
      result = TensorData.figure_dtype(4_000_000_000)
      assert_equal(:int64, result)
    end

    def test_figure_string
      result = TensorData.figure_dtype("I am a string")
      assert_equal(:string, result)
    end

    def test_figure_float
      result = TensorData.figure_dtype(32.0)
      assert_equal(:float, result)
    end

    def test_figure_double
      result = TensorData.figure_dtype(4e39)
      assert_equal(:double, result)
    end

    def test_figure_complex64
      result = TensorData.figure_dtype(Complex(4,42))
      assert_equal(:complex64, result)
    end

    def test_figure_complex128
      result = TensorData.figure_dtype(Complex(4e40,42))
      assert_equal(:complex128, result)
    end

    def test_figure_true
      result = TensorData.figure_dtype(true)
      assert_equal(:bool, result)
    end

    def test_figure_false
      result = TensorData.figure_dtype(false)
      assert_equal(:bool, result)
    end

    def test_figure_numo_int32
      result = TensorData.figure_dtype(Numo::NArray[1,2])
      assert_equal(:int32, result)
    end

    def test_figure_numo_bool
      result = TensorData.figure_dtype(Numo::NArray[true, false])
      assert_equal(:bool, result)
    end

    def test_figure_numo_string
      result = TensorData.figure_dtype(Numo::NArray["hi", "bye"])
      assert_equal(:string, result)
    end

    def test_figure_numo_tensor
      skip
      result = TensorData.figure_dtype(Numo::NArray[tensor_data.new(4), tensor_data.new(5)])
      assert_equal(:string, result)
    end

    def test_float
      value = 1.0
      tensor_data = TensorData.new(value)
      assert_equal(:float, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_float_array
      value = Numo::NArray[2.5, 3.5]
      tensor_data = TensorData.new(value, dtype: :float)
      assert_equal(:float, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_double
      value = 1.0
      tensor_data = TensorData.new(value, dtype: :double)
      assert_equal(:double, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_double_array
      value = Numo::NArray[2.5, 3.5]
      tensor_data = TensorData.new(value)
      assert_equal(:double, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_true
      value = true
      tensor_data = TensorData.new(value)
      assert_equal(:bool, tensor_data.dtype)
      assert(tensor_data.value)
    end

    def test_false
      value = false
      tensor_data = TensorData.new(value)
      assert_equal(:bool, tensor_data.dtype)
      assert_equal(0, tensor_data.value)
    end

    def test_boolean_array
      value = Numo::NArray[[true, false], [false, true]]
      tensor_data = TensorData.new(value)
      assert_equal(:bool, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_int32
      value = 1
      tensor_data = TensorData.new(value)
      assert_equal(:int32, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_int32_2
      value = 2147483647
      tensor_data = TensorData.new(value)
      assert_equal(:int32, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_int32_array
      value = Numo::NArray[[1,2,3], [3,4,5]]
      tensor_data = TensorData.new(value)
      assert_equal([2, 3], tensor_data.value.shape)
      assert_equal(:int32, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_int64
      value = 2147483648
      tensor_data = TensorData.new(value)
      assert_equal(:int64, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_int64_negative
      value = -2147483649
      tensor_data = TensorData.new(value)
      assert_equal(:int64, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_complex_64
      value = Complex(2, 3)
      tensor_data = TensorData.new(value, :dtype => :complex64)
      assert_equal(:complex64, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_complex_128
      value = Complex(2e49, 3)
      tensor_data = TensorData.new(value)
      assert_equal(:complex128, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_string
      value = "hello"
      tensor_data = TensorData.new(value)
      assert_equal(:string, tensor_data.dtype)
      assert_equal(value, tensor_data.value)
    end

    def test_string_array
      value = Numo::NArray["Hi", "there"]
      tensor_data = TensorData.new(value)
      assert_equal(:string, tensor_data.dtype)
      assert_equal(value.to_a, tensor_data.value)
    end
  end
end
