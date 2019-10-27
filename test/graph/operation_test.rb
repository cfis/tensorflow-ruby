require_relative "../test_helper"

module Tensorflow
  module Graph
    class OperationTest < Minitest::Test
      def graph
        @graph ||= Graph.new
      end

      def test_operations
        op_defs = Operation.op_defs
        assert_kind_of(Hash, op_defs)
        assert(op_defs.keys.length > 1000)
      end

      def test_op_def
        op_def = Operation.op_def('ZipDataset')
        refute_nil(op_def)
      end

      def test_name
        operation = graph.placeholder('feed')
        assert_equal('feed', operation.name)
      end

      def test_op_type
        operation = graph.placeholder
        assert_equal('Placeholder', operation.op_type)
      end

      def test_device
        operation = graph.placeholder
        assert_empty(operation.device)
      end

      def test_num_outputs
        operation = graph.placeholder
        assert_equal(1, operation.num_outputs)
      end

      def test_output_type
        operation = graph.placeholder
        assert_equal(:int32, operation.output_type)
      end

      def test_output_list_length
        operation = graph.placeholder
        assert_equal(1, operation.output_list_length('output'))
      end

      def test_consumers
        placeholder = graph.placeholder
        consumers = placeholder.consumers
        assert_empty(consumers)

        constant = graph.constant(3)
        consumers = placeholder.consumers
        assert_empty(consumers)

        add = graph.create_operation('Add', 'add') do |op_desc|
          op_desc.add_input(placeholder)
          op_desc.add_input(constant)
        end

        consumers = add.consumers
        assert_empty(consumers)

        consumers = placeholder.consumers
        assert_equal(1, consumers.length)
        assert_equal(add, consumers[0])

        consumers = constant.consumers
        assert_equal(1, consumers.length)
        assert_equal(add, consumers[0])
      end
    end
  end
end