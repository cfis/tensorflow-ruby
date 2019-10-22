require_relative "../test_helper"

module Tensorflow
  module Graph
    class OperationTest < Minitest::Test
      def test_operations
        op_defs = Operation.op_defs
        assert_kind_of(Hash, op_defs)
        assert(op_defs.keys.length > 1000)
      end

      def test_op_def
        op_def = Operation.op_def('ZipDataset')
        refute_nil(op_def)
      end

      def create_placeholder
        graph = Graph.new
        op_desc = OperationDescription.new(graph, 'Placeholder', 'feed')
        op_desc.attr('dtype').dtype = :int32
        op_desc.save
      end

      def test_name
        operation = create_placeholder
        assert_equal('feed', operation.name)
      end

      def test_op_type
        operation = create_placeholder
        assert_equal('Placeholder', operation.op_type)
      end

      def test_device
        operation = create_placeholder
        assert_empty(operation.device)
      end

      def test_num_outputs
        operation = create_placeholder
        assert_equal(1, operation.num_outputs)
      end

      def test_output_type
        operation = create_placeholder
        assert_equal(:int32, operation.output_type)
      end

      def test_output_list_length
        operation = create_placeholder
        assert_equal(1, operation.output_list_length('output'))
      end
    end
  end
end
