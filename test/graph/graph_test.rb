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
        assert_empty(graph.operations.to_a)
      end

      def test_forward
        graph = Graph.new
        a = graph.constant(1.0, 'a')
        b = graph.constant(2.1, 'b')
        c = graph.constant(2.0, 'c')
        d = Math.pow(a, c)
        y = Math.add(d, b)
        z = Math.sin(y)

        path = graph.forward(a)
        assert_equal([a, d, y, z], path.to_a)

        path = graph.forward(b)
        assert_equal([b, y, z], path.to_a)

        path = graph.forward(c)
        assert_equal([c, d, y, z], path.to_a)

        path = graph.forward(d)
        assert_equal([d, y, z], path.to_a)

        path = graph.forward(y)
        assert_equal([y, z], path.to_a)

        path = graph.forward(z)
        assert_equal([z], path.to_a)
      end

      def test_backward
        graph = Graph.new
        a = graph.constant(1.0, 'a')
        b = graph.constant(2.1, 'b')
        c = graph.constant(2.0, 'c')
        d = Math.pow(a, c)
        y = Math.add(d, b)
        z = Math.sin(y)

        path = graph.backward(z)
        assert_equal([z, y, d, a, c, b], path.to_a)

        path = graph.backward(y)
        assert_equal([y, d, a, c, b], path.to_a)

        path = graph.backward(d)
        assert_equal([d, a, c], path.to_a)

        path = graph.backward(c)
        assert_equal([c], path.to_a)

        path = graph.backward(b)
        assert_equal([b], path.to_a)

        path = graph.backward(a)
        assert_equal([a], path.to_a)
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

        placeholder = graph.placeholder('placeholder_1')
        attr = placeholder.attr('shape')
        puts attr.value
        constant = graph.constant(2, 'const_1')
        addn = Math.add_n([placeholder, constant])

        operations = graph.operations
        assert_equal(3, operations.count)

        operations = graph.operations.to_a
        assert_equal(3, operations.count)

        operation = operations[0]
        assert_equal('Placeholder', operation.op_type)
        assert_equal('placeholder_1', operation.name)

        operation = operations[1]
        assert_equal('Const', operation.op_type)
        assert_equal('const_1', operation.name)

        operation = operations[2]
        assert_equal('AddN', operation.op_type)
        assert_equal('AddN', operation.name)
      end

      def test_graph_def
        graph = Graph.new

        args_0 = graph.placeholder("args_0", :int32)
        square1 = Math.square(args_0)

        graph_def = graph.export
        puts graph_def.node
      end

      def test_export_import
        graph = Graph.new
        placeholder = graph.placeholder('feed')
        const = graph.constant(3, 'scalar')
        op_desc = OperationDescription.new(graph, 'Neg', [], name: 'neg')
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
