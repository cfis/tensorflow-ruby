require_relative "../test_helper"

module Tensorflow
  module Graph
    class GraphTest < Minitest::Test
      def test_create
        graph = Graph.new
        assert(graph)
        graph = nil
      end

      def test_op_def
        graph = Graph.new
        op_def = graph.op_def('Variable')
        assert(op_def)
        assert_equal('Variable', op_def.name)
      end

      def test_operations
        graph = Graph.new
        assert_empty(graph.operations)
      end

      def test_tensor_num_dimensions
        graph = Graph.new
        op_desc = OperationDescription.new(graph, 'Placeholder', 'arg')
        op_desc.attr('dtype').dtype = :int32
        placeholder = op_desc.save

        dims = graph.tensor_num_dims(placeholder)
        assert_equal(-1, dims)
      end

      def test_tensor_get_shape
        graph = Graph.new
        op_desc = OperationDescription.new(graph, 'Placeholder', 'arg')
        op_desc.attr('dtype').dtype = :int32
        placeholder = op_desc.save

        shape = graph.tensor_get_shape(placeholder)
        assert_equal([-1], shape)
      end

      def test_tensor_set_shape
        graph = Graph.new
        op_desc = OperationDescription.new(graph, 'Placeholder', 'arg')
        op_desc.attr('dtype').dtype = :int32
        placeholder = op_desc.save

        graph.tensor_set_shape(placeholder, [2, -1])
        dims = graph.tensor_num_dims(placeholder)
        assert_equal(2, dims)

        shape = graph.tensor_get_shape(placeholder)
        assert_equal([2, -1], shape)
      end

      def test_make_graph
        status = Status.new
        graph = Graph.new

        # Add placeholder
        op_desc = OperationDescription.new(graph, 'Placeholder', 'placeholder')
        op_desc.attr('dtype').dtype = :int32
        placeholder = op_desc.save

        # Add constant
        tensor = Tensor.new(2)
        op_desc = OperationDescription.new(graph, 'Const', 'const')
        op_desc.attr('dtype').dtype = tensor.dtype
        op_desc.attr('value').tensor = tensor
        constant = op_desc.save

        # Add add operation
        op_desc = OperationDescription.new(graph, "AddN", 'addn')
        op_desc.add_inputs(placeholder, constant)
        addn = op_desc.save

        operations = graph.operations
        assert_equal(4, operations.count)

        operation = operations[0]
        assert_equal('Placeholder', operation.op_type)
        assert_equal('placeholder', operation.name)

        operation = operations[1]
        assert_equal('Placeholder', operation.op_type)
        assert_equal('placeholder', operation.name)

        operation = operations[2]
        assert_equal('Const', operation.op_type)
        assert_equal('const', operation.name)

        operation = operations[3]
        assert_equal('AddN', operation.op_type)
        assert_equal('addn', operation.name)

        operation = graph.operation('addn')
        refute_nil(operation)
      end
    end
  end
end
