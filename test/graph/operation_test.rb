require_relative "../test_helper"

module Tensorflow
  module Graph
    class OperationTest < Minitest::Test
      def setup
        Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
        Graph.reset_default
      end

      def test_name
        operation = Tensorflow.placeholder(:int32, name: 'feed')
        assert_equal('feed', operation.name)
      end

      def test_op_type
        operation = Tensorflow.placeholder(:int32)
        assert_equal('Placeholder', operation.op_type)
      end

      def test_device
        operation = Tensorflow.placeholder(:int32)
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

      def test_attr_proto
        const = Tensorflow.constant([1,2,3])
        attr_proto = const.attr('value').proto
        assert_kind_of(Tensorflow::AttrValue, attr_proto)
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
        operation = Tensorflow.placeholder(:int32)
        assert_equal(1, operation.num_outputs)
      end

      def test_output_types
        operation = Tensorflow.placeholder(:int32)
        assert_equal([:int32], operation.output_types)
      end

      def test_dtype
        operation = Tensorflow.placeholder(:int32)
        assert_equal(:int32, operation.dtype)
      end

      def test_output_list_length
        operation = Tensorflow.placeholder(:int32)
        assert_equal(1, operation.output_list_length('output'))
      end

      def test_output_shapes
        operation = Tensorflow.placeholder(:int32, shape: [12, 32])
        assert_equal([[12, 32]], operation.output_shapes)
      end

      def test_shape
        operation = Tensorflow.placeholder(:int32, shape: [12, 32])
        assert_equal([12, 32], operation.shape)
      end

      def test_consumers
        placeholder = Tensorflow.placeholder(:int32)
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
        consumer_operation = consumers[0].operation(Graph.default)
        assert_equal(add, consumer_operation)

        consumers = constant.consumers
        assert_equal(1, consumers.length)
        consumer_operation = consumers[0].operation(Graph.default)
        assert_equal(add, consumer_operation)
      end

      def test_multiple_consumers
        data = Numo::NArray[[2,2], [2,2]]
        split = Tensorflow.split(data, 0, num_split: 2)
        rank = Tensorflow.rank(split)

        pack = rank.inputs.first.operation(Graph.default)

        consumers = split.consumers
        assert_equal(2, consumers.length)

        consumer = consumers[0]
        consumer_operation = consumer.operation(Graph.default)
        assert_equal(pack, consumer_operation)
        assert_equal(0, consumer[:index])

        consumer = consumers[1]
        consumer_operation = consumer.operation(Graph.default)
        assert_equal(pack, consumer_operation)
        assert_equal(1, consumer[:index])
      end

      def test_partial_consumers
        data = Numo::NArray[[2,2], [2,2]]
        split = Tensorflow.split(data, 0, num_split: 2)
        rank = Tensorflow.rank(split[1])

        pack = rank.inputs.first.operation(Graph.default)

        consumers = split.consumers
        assert_equal(1, consumers.length)

        consumer = consumers[0]
        consumer_operation = consumer.operation(Graph.default)
        assert_equal(rank, consumer_operation)
        assert_equal(0, consumer[:index])
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

      def test_index
        data = Numo::NArray[[2,2], [2,2]]
        split = Tensorflow.split(data, 0, num_split: 2)
        op_1 = split[1]
        assert_kind_of(FFI::Output, op_1)
        assert_equal(split, op_1.operation(split.graph))
        assert_equal(1, op_1[:index])
      end
    end
  end
end