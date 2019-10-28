require_relative "test_helper"

module Tensorflow
  class TensorflowTest < Minitest::Test
    def test_version
      assert_equal "2.0.0", Tensorflow.library_version
    end

    def test_operations
      op_defs = Tensorflow.op_defs
      assert_kind_of(Hash, op_defs)
      assert(op_defs.keys.length > 1000)
    end

    def test_op_def
      op_def = Tensorflow.op_def('ZipDataset')
      refute_nil(op_def)
    end

    # def test_fizzbuzz
    #   ret = []
    #   15.times do |i|
    #     num = Tensor.new(i + 1)
    #     if (num % 3).value == 0 && (num % 5).value == 0
    #       ret << "FizzBuzz"
    #     elsif (num % 3).value == 0
    #       ret << "Fizz"
    #     elsif (num % 5).value == 0
    #       ret << "Buzz"
    #     else
    #       ret << num.value
    #     end
    #   end
    #   assert_equal [1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"], ret
    # end
    #
    # def test_numo
    #   Tensorflow::Utils::NUMO_TYPE_MAP.each do |type, klass|
    #     value =
    #       case type
    #       when :float, :double
    #         [2.5, 3.5]
    #       else
    #         [1, 2]
    #       end
    #
    #     a = klass.cast(value)
    #     assert_equal a, Tf.identity(a).numo
    #   end
    # end
   end
end