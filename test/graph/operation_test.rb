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

      def test_node_def
        graph = Graph.new
        x = graph.constant(3.0, name: 'x')
        node_def = x.node_def
        assert(node_def)
      end

      def test_attributes
        graph = Graph.new
        operation = graph.constant(4, name: 'test')
        attributes = operation.attributes
        assert_equal(2, attributes.length)

        attribute = attributes[0]
        assert_equal('value', attribute.name)
        assert_kind_of(Tensor, attribute.value)
        assert_equal(4, attribute.value.value)

        attribute = attributes[1]
        assert_equal('dtype', attribute.name)
        assert_equal(:int32, attribute.value)
      end

      def test_num_inputs
        graph = Graph.new
        x = graph.constant(3.0, name: 'x')
        pow = Math.pow(x, 2.0)
        assert_equal(2, pow.num_inputs)
      end

      def test_inputs
        graph = Graph.new
        x = graph.constant(3.0, name: 'x')
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

      def test_add
        x = self.graph.constant(7)
        y = x + 3

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, [y])
        assert_equal(10, result)
      end

      def test_subtract
        x = self.graph.constant(7)
        y = x - 3

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, [y])
        assert_equal(4, result)
      end

      def test_multiply
        x = self.graph.constant(7)
        y = x * 3

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, [y])
        assert_equal(21, result)
      end

      def test_divide
        x = self.graph.constant(9)
        y = x / 3

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, [y])
        assert_equal(3, result)
      end

      def test_negative
        x = self.graph.constant(9)
        y = -x

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, [y])
        assert_equal(-9, result)
      end

      def test_exponent
        x = self.graph.constant(9)
        y = x ** 3

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, [y])
        assert_equal(729, result)
      end

      def test_modulus
        x = self.graph.constant(9)
        y = x % 7

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, [y])
        assert_equal(2, result)
      end
    end
  end
end