require_relative "../test_helper"

module Tensorflow
  class MathTest < Minitest::Test
    def setup
      Tensorflow.execution_mode = Tensorflow::EAGER_MODE
    end

    def test_abs
      assert_equal(1, Tensorflow.abs(Tensor.new(-1)).value)
      assert_equal(1, Tensorflow::Math.abs(Tensor.new(-1)).value)
    end

    def test_add
      a = Tensor.new(2)
      b = Tensor.new(3)
      assert_equal(5, (a + b).value)
      assert_equal(5, Tensorflow.add(a, b).value)
      assert_equal(5, Tensorflow::Math.add(a, b).value)
    end

    def test_divide
      a = Tensor.new(3.0)
      b = Tensor.new(2.0)
      assert_in_delta(1.5, (a / b).value)
      assert_in_delta(1.5, Tensorflow.divide(a, b).value)
      assert_in_delta(1.5, Tensorflow::Math.divide(a, b).value)
    end

    def test_equal
      x = Tensor.new([2, 4])
      y = Tensor.new(2)
      assert_equal([1, 0], Tensorflow.equal(x, y).value)
      assert_equal([1, 0], Tensorflow::Math.equal(x, y).value)

      x = Tensor.new([2, 4])
      y = Tensor.new([2, 4])
      assert_equal([1, 1], Tensorflow.equal(x, y).value)
      assert_equal([1, 1], Tensorflow::Math.equal(x, y).value)
    end

    def test_logical_xor
      x = Tensor.new([false, false, true, true])
      y = Tensor.new([false, true, false, true])
      assert_equal([0, 1, 1, 0], Tensorflow::Math.logical_xor(x, y).value)
    end

    def test_log_sigmoid
      assert_in_delta(-0.31326166, Tensorflow::Math.log_sigmoid(1.0).value)
    end

    def test_multiply
      a = Tensor.new(2)
      b = Tensor.new(3)
      assert_equal(6, (a * b).value)
      assert_equal(6, Tensorflow.multiply(a, b).value)
      assert_equal(6, Tensorflow::Math.multiply(a, b).value)

      a = Tensor.new([[1, 2], [3, 4]])
      b = Tensorflow.add(a, 1)
      assert_equal([[2, 6], [12, 20]], (a * b).value)
    end

    def test_negative
      assert_equal([-1, -2], Tensorflow.negative([1, 2]).value)
    end

    def test_reduce_any
      x = Tensor.new([[true, true], [false, false]])
      assert_equal(1, Tensorflow.reduce_any(x).value)
      assert_equal([1, 1], Tensorflow.reduce_any(x, axis: 0).value)
      assert_equal([1, 0], Tensorflow.reduce_any(x, axis: 1).value)
    end

    def test_reduce_max
      x = Tensor.new([[1, 2], [3, 4]])
      assert_equal(4, Tensorflow.reduce_max(x).value)
      assert_equal([3, 4], Tensorflow.reduce_max(x, axis: 0).value)
      assert_equal([2, 4], Tensorflow.reduce_max(x, axis: 1).value)
    end

    def test_reduce_mean_constant
      x = Tensor.new([[1.0, 1.0], [2.0, 2.0]])
      assert_equal(1.5, Tensorflow.reduce_mean(x).value)
      assert_equal([1.5, 1.5], Tensorflow.reduce_mean(x, axis: 0).value)
      assert_equal([1.0, 2.0], Tensorflow.reduce_mean(x, axis: 1).value)
    end

    def test_reduce_mean_variable
      x = Tensorflow::Variable.new([[1.0, 1.0], [2.0, 2.0]])
      assert_equal(1.5, Tensorflow.reduce_mean(x).value)
      assert_equal([1.5, 1.5], Tensorflow.reduce_mean(x, axis: 0).value)
      assert_equal([1.0, 2.0], Tensorflow.reduce_mean(x, axis: 1).value)
    end

    def test_reduce_min
      x = Tensor.new([[1, 2], [3, 4]])
      assert_equal(1, Tensorflow.reduce_min(x).value)
      assert_equal([1, 2], Tensorflow.reduce_min(x, axis: 0).value)
      assert_equal([1, 3], Tensorflow.reduce_min(x, axis: 1).value)
    end

    def test_reduce_prod
      x = Tensor.new([[1, 2], [3, 4]])
      assert_equal(24, Tensorflow.reduce_prod(x).value)
      assert_equal([3, 8], Tensorflow.reduce_prod(x, axis: 0).value)
      assert_equal([2, 12], Tensorflow.reduce_prod(x, axis: 1).value)
    end

    def test_reduce_std
      x = Tensor.new([[1.0, 2.0], [3.0, 4.0]])
      assert_in_delta(1.1180339887498949, Tensorflow::Math.reduce_std(x).value)
      assert_equal([1.0, 1.0], Tensorflow::Math.reduce_std(x, axis: 0).value)
      assert_equal([0.5, 0.5], Tensorflow::Math.reduce_std(x, axis: 1).value)
    end

    def test_reduce_sum
      x = Tensor.new([[1, 1, 1], [1, 1, 1]])
      assert_equal(6, Tensorflow.reduce_sum(x).value)
      assert_equal([2, 2, 2], Tensorflow.reduce_sum(x, axis: 0).value)
      assert_equal([3, 3], Tensorflow.reduce_sum(x, axis: 1).value)
      assert_equal([[3], [3]], Tensorflow.reduce_sum(x, axis: 1, keepdims: true).value)
      assert_equal(6, Tensorflow.reduce_sum(x, axis: [0, 1]).value)
    end

    def test_reduce_sum_graph
      Graph::Graph.new.as_default do |graph|
        x = Tensorflow.constant([[1, 1, 1], [1, 1, 1]])

        session = Graph::Session.new(graph, Graph::SessionOptions.new)

        y = Tensorflow.reduce_sum(x)
        result = session.run([y])
        assert_equal(6, result)

        y = Tensorflow.reduce_sum(x, axis: 0)
        result = session.run([y])
        assert_equal([2, 2, 2], result)

        y = Tensorflow.reduce_sum(x, axis: 1)
        result = session.run([y])
        assert_equal([3, 3], result)

        y = Tensorflow.reduce_sum(x, axis: 1, keepdims: true)
        result = session.run([y])
        assert_equal([[3], [3]], result.to_a)

        y = Tensorflow.reduce_sum(x, axis: [0, 1])
        result = session.run([y])
        assert_equal(6, result)
      end
    end

    def test_reduce_variance
      x = Tensor.new([[1.0, 2.0], [3.0, 4.0]])
      assert_equal(1.25, Tensorflow::Math.reduce_variance(x).value)
      assert_equal([1, 1], Tensorflow::Math.reduce_variance(x, axis: 0).value.to_a)
      assert_equal([0.25, 0.25], Tensorflow::Math.reduce_variance(x, axis: 1).value)
    end

    def test_sin
      assert_equal([0, 1], Tensorflow.sin([0.0, 0.5 * ::Math::PI]).value.to_a)
      assert_equal([0, 1], Tensorflow::Math.sin([0.0, 0.5 * ::Math::PI]).value.to_a)
    end

    def test_sqrt
      assert_equal([2.0, 3.0], Tensorflow.sqrt([4.0, 9.0]).value)
      assert_equal([2.0, 3.0], Tensorflow::Math.sqrt([4.0, 9.0]).value)
    end

    def test_subtract
      a = Tensor.new(2)
      b = Tensor.new(3)
      assert_equal(-1, (a - b).value)
      assert_equal(-1, Tensorflow.subtract(a, b).value)
      assert_equal(-1, Tensorflow::Math.subtract(a, b).value)
    end
  end
end