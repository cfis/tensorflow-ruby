require_relative "../base_test"

module Tensorflow
  class RandomTest < BaseTest
    def test_normal
      self.eager_and_graph do |context|
        op = Random.normal([2, 2], stddev: 10.0)
        assert_equal([2, 2], op.shape)
        assert_equal(:float, op.dtype)
      end
    end

    def test_normal_different_eager
      Eager::Context.new.as_default do |context|
        op_1 = Random.normal([2, 2], stddev: 10.0)
        result_1 = self.evaluate(op_1)

        op_2 = Random.normal([2, 2], stddev: 10.0)
        result_2 = self.evaluate(op_2)
        refute_equal(result_1, result_2)
      end
    end

    def test_normal_same_eager
      Eager::Context.new.as_default do |context|
        op_1 = Random.normal([2, 2], seed: 55)
        result_1 = self.evaluate(op_1)

        op_2 = Random.normal([2, 2], seed: 55)
        result_2 = self.evaluate(op_2)

        # Seems like seeds don't matter in eager mode??
        refute_equal(result_1, result_2)
      end
    end

    def test_normal_different_graph
      result_1 = nil
      result_2 = nil

      self.graph_mode do |context|
        op = Random.normal([2, 2], stddev: 10.0)
        result_1 = self.evaluate(op)
      end

      self.graph_mode do |context|
        op = Random.normal([2, 2], stddev: 10.0)
        result_2 = self.evaluate(op)
      end
      refute_equal(result_1, result_2)
    end

    def test_normal_same_graph
      self.graph_mode do |context|
        op = Random.normal([2, 2], seed: 55)
        result_1 = self.evaluate(op)
        result_2 = self.evaluate(op)
        refute_equal(result_1, result_2)
      end
    end

    def test_uniform
      self.eager_and_graph do |context|
        op = Random.uniform([2, 2])
        assert_equal([2, 2], op.shape)
        assert_equal(:float, op.dtype)
      end
    end

    def test_truncated_normal
      self.graph_mode do |context|
        op = Random.truncated_normal([2, 2], seed: 10.0)
        puts self.evaluate(op)
      end
    end
  end
end