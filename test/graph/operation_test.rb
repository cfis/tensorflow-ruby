require_relative "../test_helper"

module Tensorflow
  module Graph
    class OperationTest < Minitest::Test
      def setup
        Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
      end

      def test_name
        operation = Tensorflow.placeholder('feed')
        assert_equal('feed', operation.name)
      end

      def test_op_type
        operation = Tensorflow.placeholder
        assert_equal('Placeholder', operation.op_type)
      end

      def test_device
        operation = Tensorflow.placeholder
        assert_empty(operation.device)
      end

      def test_node_def
        graph = Graph.new
        x = Tensorflow.constant(3.0, name: 'x')
        node_def = x.node_def
        assert(node_def)
      end

      def test_attributes
        operation = Tensorflow.constant(4, name: 'test')
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
        x = Tensorflow.constant(3.0, name: 'x')
        pow = Math.pow(x, 2.0)
        assert_equal(2, pow.num_inputs)
      end

      def test_inputs
        x = Tensorflow.constant(3.0, name: 'x')
        pow = Math.pow(x, 2.0)

        inputs = pow.inputs
        assert_equal(2, inputs.length)
        operation = inputs[0].operation(x.graph)
        assert_equal(x, operation)

        powy = x.graph.operation('Pow/y')
        operation = inputs[1].operation(x.graph)
        assert_equal(powy, operation)
      end

      def test_num_outputs
        operation = Tensorflow.placeholder
        assert_equal(1, operation.num_outputs)
      end

      def test_output_types
        operation = Tensorflow.placeholder
        assert_equal([:int32], operation.output_types)
      end

      def test_output_list_length
        operation = Tensorflow.placeholder
        assert_equal(1, operation.output_list_length('output'))
      end

      def test_consumers
        placeholder = Tensorflow.placeholder
        consumers = placeholder.consumers
        assert_empty(consumers)

        constant = Tensorflow.constant(3)
        consumers = placeholder.consumers
        assert_empty(consumers)

        add = Math.add(placeholder, constant)
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
        x = Tensorflow.constant(7)
        y = x + 3

        session = Session.new(x.graph, SessionOptions.new)
        result = session.run([y])
        assert_equal(10, result)
      end

      def test_subtract
        x = Tensorflow.constant(7)
        y = x - 3

        session = Session.new(x.graph, SessionOptions.new)
        result = session.run([y])
        assert_equal(4, result)
      end

      def test_multiply
        x = Tensorflow.constant(7)
        y = x * 3

        session = Session.new(x.graph, SessionOptions.new)
        result = session.run([y])
        assert_equal(21, result)
      end

      def test_divide
        x = Tensorflow.constant(9)
        y = x / 3

        session = Session.new(x.graph, SessionOptions.new)
        result = session.run([y])
        assert_equal(3, result)
      end

      def test_negative
        x = Tensorflow.constant(9)
        y = -x

        session = Session.new(x.graph, SessionOptions.new)
        result = session.run([y])
        assert_equal(-9, result)
      end

      def test_exponent
        x = Tensorflow.constant(9)
        y = x ** 3

        session = Session.new(x.graph, SessionOptions.new)
        result = session.run([y])
        assert_equal(729, result)
      end

      def test_modulus
        x = Tensorflow.constant(9)
        y = x % 7

        session = Session.new(x.graph, SessionOptions.new)
        result = session.run([y])
        assert_equal(2, result)
      end
    end
  end
end