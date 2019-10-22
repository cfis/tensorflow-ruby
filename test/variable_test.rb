require_relative "test_helper"

module Tensorflow
  class VariableTest < Minitest::Test
    def test_value
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      assert_equal([[[0, 1, 2],
                     [3, 4, 5]],
                    [[6, 7, 8],
                     [9, 10, 11]]], var1.value)
    end

    def test_tensor
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      tensor = var1.tensor
      assert_equal([[[0, 1, 2],
                     [3, 4, 5]],
                    [[6, 7, 8],
                     [9, 10, 11]]], tensor.value)
    end

    def test_rank
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      assert_equal(3, var1.rank)
    end

    def test_dtype
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      assert_equal(:int32, var1.dtype)
    end

    def test_shape
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])
      assert_equal([2, 2, 3], var1.shape)
    end

    def test_reshape
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      tensor = var1.reshape([2, 6])
      assert_equal(:int32, tensor.dtype)
      assert_equal([2, 6], tensor.shape)
      assert_equal([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], tensor.value)
    end

    def test_addition
      v = Variable.new(0.0)
      w = v + 1
      assert_equal(0.0, v.value)
      assert_equal(1.0, w.value)
    end

      def test_subtraction
        v = Variable.new(0.0)
        x = v - 1
        assert_equal(0.0, v.value)
        assert_equal(-1.0, x.value)
      end
  end
end