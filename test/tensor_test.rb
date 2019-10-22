require_relative "test_helper"

module Tensorflow
  class TensorTest < Minitest::Test
    def test_infer_type
      assert_equal(:float, Tensor.new(1.0).dtype)
      assert_equal(:float, Tensor.new(1.234567890123456789).dtype)
      assert_equal(:bool, Tensor.new([true, false]).dtype)
      assert_equal(:int32, Tensor.new(1).dtype)
      assert_equal(:int32, Tensor.new(2147483647).dtype)
      assert_equal(:int32, Tensor.new(-2147483648).dtype)
      assert_equal(:int64, Tensor.new(2147483648).dtype)
      assert_equal(:int64, Tensor.new(-2147483649).dtype)
      assert_equal(:complex128, Tensor.new(Complex(2, 3)).dtype)
      assert_equal(:string, Tensor.new(["hello", "world"]).dtype)
    end

    def test_narray_1_d
      narray = Numo::NArray[1, 2, 3, 4]
      tensor = Tensor.new(narray)
      assert_equal([4], tensor.shape)
      assert_equal(:int32, tensor.dtype)
      assert_equal([1, 2, 3, 4], tensor.value)
    end

    def test_narray_2_d
      narray = Numo::NArray[[1], [2], [3], [4]]
      tensor = Tensor.new(narray)
      assert_equal([4, 1], tensor.shape)
      assert_equal(:int32, tensor.dtype)
      assert_equal([[1], [2], [3], [4]], tensor.value)
    end

    def test_string
      value = "I am a string"
      tensor = Tensor.new(value)
      puts tensor
      assert_equal([], tensor.shape)
      assert_equal(:string, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    def test_integer
      value = 9
      tensor = Tensor.new(value)
      assert_equal([], tensor.shape)
      assert_equal(:int32, tensor.dtype)
      assert_equal(value, tensor.value)
    end

    # def test_const_integer
    #   value = 9
    #   tensor = Tensor.constant(value)
    #   assert_equal([], tensor.shape)
    #   assert_equal(:int32, tensor.dtype)
    #   assert_equal(value, tensor.value)
    # end
    #
    # def self.constant(value, dtype: nil, shape: nil)
    #   tensor = Tensor.new(value, dtype: dtype, shape: shape)
    #   RawOps.const(value: tensor, dtype: tensor.dtype)
    # end
  end
end 
