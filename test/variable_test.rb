require_relative "test_helper"

module Tensorflow
  class VariableTest < Minitest::Test
    def setup
      Tensorflow.execution_mode = Tensorflow::EAGER_MODE
    end

    def test_simple_eager
      var = Variable.new(32)
      assert_kind_of(Eager::TensorHandle, var.handle)
      assert_equal(:int32, var.dtype)
      assert_equal([], var.shape)
      assert_equal(32, var.value)
    end

    def test_simple_graph
      Tensorflow.disable_eager_execution
      Graph::Graph.new.as_default do |graph|
        var = Variable.new(32)
        assert_kind_of(Graph::Operation, var.handle)
        assert_equal(:int32, var.dtype)
        assert_raises(TensorflowError) do
          var.shape
        end

        assert_raises(TensorflowError) do
          var.tensor
        end

        assert_raises(TensorflowError) do
          var.value
        end
      end
    end

    def test_float
      x = Tensorflow::Variable.new(1.0)
      assert_equal(1.0, x.value)

      handle = Eager::TensorHandle.from_value(x)
      assert_equal(1.0, x.value)
    end

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
      w = v + 1.0
      assert_equal(0.0, v.value)
      assert_equal(1.0, w.value)
    end

    def test_subtraction
      v = Variable.new(0.0)
      x = v - 1.0
      assert_equal(0.0, v.value)
      assert_equal(-1.0, x.value)
    end
  end
end