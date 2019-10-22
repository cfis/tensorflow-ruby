require_relative "../test_helper"

module Tensorflow
  module Eager
    class OperationTest < Minitest::Test
      def test_number_attr
        op_def = Graph::Operation.op_def('ZipDataset')
        arg_def = op_def.input_arg.first
        assert_equal('N', arg_def.number_attr)
        assert_empty(arg_def.type_list_attr)
      end

      def test_type_list_attr
        op_def = Graph::Operation.op_def('TensorSliceDataset')
        arg_def = op_def.input_arg.first
        assert_empty(arg_def.number_attr)
        assert_equal('Toutput_types', arg_def.type_list_attr)
      end
    end
  end
end
