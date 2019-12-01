require_relative "base_test"

module Tensorflow
  class ConstantTest < BaseTest
    def setup
      Tensorflow.disable_eager_execution
    end

    def test_constant_int32
      const = Tensorflow.constant(3)
      tensor = const.attr('value').tensor
      assert_equal(:int32, tensor.dtype)
      assert_equal([], tensor.shape)
      assert_equal(3, tensor.value, 0.1)
    end

    def test_constant_int64
      const = Tensorflow.constant(10_000_000_000)
      tensor = const.attr('value').tensor
      assert_equal(:int64, tensor.dtype)
      assert_equal([], tensor.shape)
      assert_equal(10_000_000_000, tensor.value, 0.1)
    end

    def test_constant_float
      const = Tensorflow.constant(3.3)
      tensor = const.attr('value').tensor
      assert_equal(:float, tensor.dtype)
      assert_equal([], tensor.shape)
      assert_in_delta(3.3, tensor.value, 0.1)
    end

    def test_constant_narray
      narray = Numo::NArray[[1,2,3], [3,4,5]]
      const = Tensorflow.constant(narray)
      tensor = const.attr('value').tensor
      assert_equal(:int32, tensor.dtype)
      assert_equal([2, 3], tensor.shape)
      assert_equal(Numo::NArray[[1,2,3], [3,4,5]], tensor.value)
    end

    def test_constant_tensor
      tensor = Tensor.new([[1,2,3], [3,4,5]])
      const = Tensorflow.constant(tensor)
      tensor = const.attr('value').tensor
      assert_equal(:int32, tensor.dtype)
      assert_equal([2, 3], tensor.shape)
      assert_equal([[1,2,3], [3,4,5]], tensor.value)
    end

    def test_constant_true
      const = Tensorflow.constant(true)
      tensor = const.attr('value').tensor
      assert_equal(:bool, tensor.dtype)
      assert_equal([], tensor.shape)
      assert(tensor.value)
    end

    def test_constant_false
      const = Tensorflow.constant(false)
      tensor = const.attr('value').tensor
      assert_equal(:bool, tensor.dtype)
      assert_equal([], tensor.shape)
      assert_equal(0, tensor.value)
    end

    def test_constant_scalar_with_shape
      const = Tensorflow.constant(3, shape: [3,2])
      tensor = const.attr('value').tensor
      assert_equal(:int32, tensor.dtype)
      assert_equal([3, 2], tensor.shape)
      assert_equal([[3, 3], [3, 3], [3, 3]], tensor.value)
    end
  end
end
