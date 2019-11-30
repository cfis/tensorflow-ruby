require_relative "../base_test"

module Tensorflow
  class OpsTest < BaseTest
    def test_const
      self.eager_and_graph do |context|
        op = Tensorflow.constant(33)
        result = self.evaluate(op)
        assert_equal(33, result)
      end
    end

    def test_eye
      self.eager_and_graph do |context|
        op = Tensorflow.eye(2)
        result = self.evaluate(op)
        assert_equal([[1.0, 0.0], [0.0, 1.0]], result)
      end
    end

    def test_fill
      self.eager_and_graph do |context|
        op = Tensorflow.fill([2, 3], 9)
        result = self.evaluate(op)
        assert_equal([[9, 9, 9], [9, 9, 9]], result)
      end
    end

    def test_identity
      [:float, :double, :int32, :uint8, :int16, :int8, :int64, :uint16, :uint32, :uint64, :bool, :string, :complex64, :complex128].each do |dtype|
        value = case dtype
                  when :string
                    ["hello", "world"]
                  when :bool
                    [1, 0]
                  when :complex64
                    [Complex(2.0, 3.0), Complex(1.0, 2.0)]
                  when :complex128
                    [Complex(2.0e52, 3.0), Complex(1.0, 2.0)]
                  when :float
                    [2.5, 3.5]
                  when :double
                    [2.5e48, 3.5e48]
                  else
                    [1, 2]
                  end

        tensor = Tensor.new(value, dtype: dtype)

        self.eager_and_graph do |context|
          op = Tensorflow.identity(tensor)
          assert_equal(dtype, tensor.dtype)
          result = self.evaluate(op)
          assert_equal(value, result)
        end
      end
    end

    def test_ones
      self.eager_and_graph do |context|
        op = Tensorflow.ones([2, 3])
        result = self.evaluate(op)
        assert_equal([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], result)

        op = Tensorflow.ones([2, 3], dtype: :int32)
        result = self.evaluate(op)
        assert_equal([[1, 1, 1], [1, 1, 1]], result)
      end
    end

    def test_pack
      x = [1, 4]
      y = [2, 5]
      z = [3, 6]

      self.eager_and_graph do |context|
        op = Tensorflow.pack([x, y, z])
        result = self.evaluate(op)
        assert_equal([[1, 4], [2, 5], [3, 6]], result)

        op = Tensorflow.pack([x, y, z], axis: 1)
        result = self.evaluate(op)
        assert_equal([[1, 2, 3], [4, 5, 6]], result)
      end

      x = Tensor.new([1, 4])
      y = Tensor.new([2, 5])
      z = Tensor.new([3, 6])

      self.eager_and_graph do |context|
        op = Tensorflow.pack([x, y, z])
        result = self.evaluate(op)
        assert_equal([[1, 4], [2, 5], [3, 6]], result)

        op = Tensorflow.pack([x, y, z], axis: 1)
        result = self.evaluate(op)
        assert_equal([[1, 2, 3], [4, 5, 6]], result)
      end
    end

    def test_placeholder
      Graph::Graph.default.as_default do |graph|
        placeholder = Tensorflow.placeholder(:int32, name: 'placeholder_1')
        assert_equal('placeholder_1', placeholder.name)
        assert_equal('Placeholder', placeholder.op_type)
        shapes = graph.output_shapes(placeholder)
        assert_equal([[]], shapes)
      end
    end

    def test_rank
      self.eager_and_graph do |context|
        op = Tensorflow.rank(1)
        result = self.evaluate(op)
        assert_equal(0, result)

        op = Tensorflow.rank([1])
        result = self.evaluate(op)
        assert_equal(1, result)

        op = Tensorflow.rank([1, 2])
        result = self.evaluate(op)
        assert_equal(1, result)

        op = Tensorflow.rank([[1]])
        result = self.evaluate(op)
        assert_equal(2, result)

        op = Tensorflow.rank([[1, 1], [2,2]])
        result = self.evaluate(op)
        assert_equal(2, result)
      end
    end

    def test_range
      self.eager_and_graph do |context|
        op = Tensorflow.range(3)
        result = self.evaluate(op)
        assert_equal([0, 1, 2], result)

        op = Tensorflow.range(3, 18, 3)
        result = self.evaluate(op)
        assert_equal([3, 6, 9, 12, 15], result)
      end
    end

    def test_split
      value = Numo::Int32.new([5, 30]).seq

      self.eager_and_graph do |context|
        op = Tensorflow.split(value, 1, num_split: 3)
        result = self.evaluate(op)
        assert_equal(3, result.length)
        assert_equal([5, 10], result[0].shape)
        assert_equal([5, 10], result[1].shape)
        assert_equal([5, 10], result[2].shape)
      end
    end

    def test_split_v
      value = Numo::Int32.new([5, 30]).seq

      self.eager_and_graph do |context|
        op = Tensorflow.split_v(value, [4, 15, 11], 1)
        result = self.evaluate(op)
        assert_equal(3, result.length)
        assert_equal([5, 4], result[0].shape)
        assert_equal([5, 15], result[1].shape)
        assert_equal([5, 11], result[2].shape)
      end
    end

    def test_timestamp
      self.eager_and_graph do |context|
        op = Tensorflow.timestamp
        result = self.evaluate(op)
        assert_in_delta(Time.now.to_f, result, 1)
      end
    end

    def test_transpose
      self.eager_and_graph do |context|
        op = Tensorflow.transpose([[1, 2, 3], [4, 5, 6]])
        result = self.evaluate(op)
        assert_equal([[1, 4], [2, 5], [3, 6]], result)
      end
    end

    def test_zeros
      self.eager_and_graph do |context|
        op = Tensorflow.zeros([2, 3])
        assert_equal(:float, op.dtype)
        result = self.evaluate(op)
        assert_equal([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], result)

        op = Tensorflow.zeros([2, 3], dtype: :int64)
        assert_equal(:int64, op.dtype)
        result = self.evaluate(op)
        assert_equal([[0, 0, 0], [0, 0, 0]], result)
      end
    end

    def test_zeros_no_shape
      self.eager_and_graph do |context|
        op = Tensorflow.zeros([], dtype: :int64)
        result = self.evaluate(op)
        assert_equal(0, result)
        assert_equal(:int64, op.dtype)
      end
    end

    def test_zeros_like
      self.eager_and_graph do |context|
        ones = Tensorflow.ones([2, 3])
        op = Tensorflow.zeros_like(ones)
        result = self.evaluate(op)
        assert_equal([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], result)
      end
    end
  end
end
