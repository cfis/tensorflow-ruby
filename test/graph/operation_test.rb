require_relative "../test_helper"

module Tensorflow
  module Graph
    class OperationTest < Minitest::Test
      def graph
        @graph ||= Graph.new
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

      def test_num_inputs
        graph = Graph.new
        x = graph.constant(3.0, 'x')
        pow = Math.pow(x, 2.0)
        assert_equal(2, pow.num_inputs)
      end

      def test_inputs
        graph = Graph.new
        x = graph.constant(3.0, 'x')
        pow = Math.pow(x, 2.0)

        inputs = pow.inputs
        assert_equal(2, inputs.length)
        assert_equal(x, inputs[0])

        powy = graph.operation('Pow/y')
        assert_equal(powy, inputs[1])
      end

      def test_num_outputs
        operation = graph.placeholder
        assert_equal(1, operation.num_outputs)
      end

      def test_output_types
        operation = graph.placeholder
        assert_equal([:int32], operation.output_types)
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

        add = graph.create_operation('Add', [placeholder, constant], name: 'add')
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