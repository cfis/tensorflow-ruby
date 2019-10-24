require_relative "../test_helper"

module Tensorflow
  module Graph
    class GraphTest < Minitest::Test
      def test_create
        graph = Graph.new
        assert(graph)
        graph = nil
      end

      def test_placeholder
        graph = Graph.new
        placeholder = graph.placeholder('placeholder_1')

        assert_equal('placeholder_1', placeholder.name)
        assert_equal('Placeholder', placeholder.op_type)
        dims = graph.tensor_num_dims(placeholder)
        assert_equal(-1, dims)
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
        placeholder = graph.placeholder('placeholder_1')
        dims = graph.tensor_num_dims(placeholder)
        assert_equal(-1, dims)
      end

      def test_tensor_get_shape
        graph = Graph.new
        placeholder = graph.placeholder('placeholder_1')
        shape = graph.tensor_get_shape(placeholder)
        assert_equal([-1], shape)
      end

      def test_tensor_set_shape
        graph = Graph.new
        placeholder = graph.placeholder('placeholder_1')
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
        placeholder = graph.placeholder('placeholder_1')

        # Add constant
        constant = graph.constant(2, 'const_1')

        # Add add operation
        op_desc = OperationDescription.new(graph, "AddN", 'addn')
        op_desc.add_input_list([placeholder, constant])
        addn = op_desc.save

        operations = graph.operations
        assert_equal(4, operations.count)

        operation = operations[0]
        assert_equal('Placeholder', operation.op_type)
        assert_equal('placeholder_1', operation.name)

        operation = operations[1]
        assert_equal('Placeholder', operation.op_type)
        assert_equal('placeholder_1', operation.name)

        operation = operations[2]
        assert_equal('Const', operation.op_type)
        assert_equal('const_1', operation.name)

        operation = operations[3]
        assert_equal('AddN', operation.op_type)
        assert_equal('addn', operation.name)

        operation = graph.operation('addn')
        refute_nil(operation)
      end

      def test_graph_def
        graph = Graph.new

        op_desc = OperationDescription.new(graph, 'Placeholder', "args_0")
        op_desc.attr('dtype').dtype = :int32
        op_desc.attr('shape').shape = [4]
        args_0 = op_desc.save

        op_desc = OperationDescription.new(graph, 'Square', 'square1')
        op_desc.add_input(args_0)
        square1 = op_desc.save

        graph_def = graph.export
        puts graph_def.node
      end

      def test_export_import
        graph = Graph.new
        placeholder = graph.placeholder('feed')
        const = graph.constant(3, 'scalar')
        op_desc = OperationDescription.new(graph, 'Neg', 'neg')
        op_desc.add_input(const)
        negate = op_desc.save

        assert(graph.operation('feed'))
        assert(graph.operation('scalar'))
        assert(graph.operation('neg'))

        # Get def
        graph_def = graph.export

        new_graph = Graph.new
        options = GraphDefOptions.new
        options.prefix = 'imported'
        new_graph.import(graph_def, options)

        assert(new_graph.operation('imported/feed'))
        assert(new_graph.operation('imported/scalar'))
        assert(new_graph.operation('imported/neg'))
      end
    end
  end
end
