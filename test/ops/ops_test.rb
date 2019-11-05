require_relative "../test_helper"

module Tensorflow
  class OpsTest < Minitest::Test
    def test_const
      const = Tf.const(33)
      assert_kind_of(Eager::TensorHandle, const)
      assert_equal(:int32, const.dtype)
      assert_equal(33, const.value)
    end

    def test_eye
      assert_equal([[1, 0], [0, 1]], Tf.eye(2).value)
    end

    def test_fill
      assert_equal([[9, 9, 9], [9, 9, 9]], Tf.fill([2, 3], 9).value)
    end

    def test_identity
      [:float, :double, :int32, :uint8, :int16, :int8, :int64, :uint16, :uint32, :uint64, :bool, :string, :complex64, :complex128].each do |dtype|
        value =
          case dtype
          when :string
            ["hello", "world"]
          when :bool
            [1, 0]
          when :complex64
            [Complex(2, 3), Complex(1, 2)]
          when :complex128
            [Complex(2e52, 3), Complex(1, 2)]
          when :float
            [2.5, 3.5]
          when :double
            [2.5e48, 3.5e48]
          else
            [1, 2]
          end

        tensor = Tf.identity(Tf::Tensor.new(value, dtype: dtype))
        assert_equal(dtype, tensor.dtype)
        assert_equal(value, tensor.value.to_a)
      end
    end

    def test_ones
      assert_equal([[1, 1, 1], [1, 1, 1]], Tf.ones([2, 3]).value)
    end

    def test_pack
      x = [1, 4]
      y = [2, 5]
      z = [3, 6]

      assert_equal([[1, 4], [2, 5], [3, 6]], Tf.pack([x, y, z]).value)
      assert_equal([[1, 2, 3], [4, 5, 6]], Tf.pack([x, y, z], axis: 1).value)

      x = Tensor.new([1, 4])
      y = Tensor.new([2, 5])
      z = Tensor.new([3, 6])

      assert_equal([[1, 4], [2, 5], [3, 6]], Tf.pack([x, y, z]).value)
      assert_equal([[1, 2, 3], [4, 5, 6]], Tf.pack([x, y, z], axis: 1).value)
    end

    def test_rank
      assert_equal(0, Tf.rank(1).value)
      assert_equal(1, Tf.rank([1]).value)
      assert_equal(1, Tf.rank([1,2]).value)
      assert_equal(2, Tf.rank([[1]]).value)
      assert_equal(2, Tf.rank([[1, 1], [2,2]]).value)
      assert_equal(2, Tf.rank([Tensor.new([1, 1]), Tensor.new([2,2])]).value)
     end

    def test_range
      assert_equal([0, 1, 2], Tf.range(3).value)
      assert_equal([3, 6, 9, 12, 15], Tf.range(3, 18, 3).value)
    end

    def test_split
      value = Numo::Int32.new([5, 30]).seq
      split0, split1, split2 = Tf.split(value, 1, num_split: 3)
      assert_equal([5, 10], split0.shape)
      assert_equal([5, 10], split1.shape)
      assert_equal([5, 10], split2.shape)
    end

    def test_split_v
      value = Numo::Int32.new([5, 30]).seq
      split0, split1, split2 = Tf.split_v(value, [4, 15, 11], 1)
      assert_equal([5, 4], split0.shape)
      assert_equal([5, 15], split1.shape)
      assert_equal([5, 11], split2.shape)
    end

    def test_timestamp
      assert_in_delta Time.now.to_f, Tf.timestamp.value, 1
    end

    def test_transpose
      assert_equal([[1, 4], [2, 5], [3, 6]], Tf.transpose([[1, 2, 3], [4, 5, 6]]).value)
    end

    def test_zeros
      assert_equal([[0, 0, 0], [0, 0, 0]], Tf.zeros([2, 3]).value)
    end

    def test_zeros_like
      assert_equal([[0, 0, 0], [0, 0, 0]], Tf.zeros_like(Tf.ones([2, 3])).value)
    end
  end
end