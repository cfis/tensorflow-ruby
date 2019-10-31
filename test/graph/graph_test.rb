require_relative "../test_helper"

module Tensorflow
  module Graph
    class GraphTest < Minitest::Test
      def graph
        @graph ||= Graph.new
      end

      def test_create
        assert(self.graph)
      end

      def test_placeholder
        placeholder = self.graph.placeholder('placeholder_1')

        assert_equal('placeholder_1', placeholder.name)
        assert_equal('Placeholder', placeholder.op_type)
        dims = self.graph.tensor_num_dims(placeholder)
        assert_equal(-1, dims)
      end

      def test_op_def
        op_def = self.graph.op_def('Variable')
        assert(op_def)
        assert_equal('Variable', op_def.name)
      end

      def test_operations
        assert_empty(self.graph.operations.to_a)
      end

      def test_forward
        a = self.graph.constant(1.0, name: 'a')
        b = self.graph.constant(2.1, name: 'b')
        c = self.graph.constant(2.0, name: 'c')
        d = Math.pow(a, c)
        y = Math.add(d, b)
        z = Math.sin(y)

        path = self.graph.forward(a)
        assert_equal([a, d, y, z], path.to_a)

        path = self.graph.forward(b)
        assert_equal([b, y, z], path.to_a)

        path = self.graph.forward(c)
        assert_equal([c, d, y, z], path.to_a)

        path = self.graph.forward(d)
        assert_equal([d, y, z], path.to_a)

        path = self.graph.forward(y)
        assert_equal([y, z], path.to_a)

        path = self.graph.forward(z)
        assert_equal([z], path.to_a)
      end

      def test_backward
        a = self.graph.constant(1.0, name: 'a')
        b = self.graph.constant(2.1, name: 'b')
        c = self.graph.constant(2.0, name: 'c')
        d = Math.pow(a, c)
        y = Math.add(d, b)
        z = Math.sin(y)

        path = self.graph.backward(z)
        assert_equal([z, y, d, a, c, b], path.to_a)

        path = self.graph.backward(y)
        assert_equal([y, d, a, c, b], path.to_a)

        path = self.graph.backward(d)
        assert_equal([d, a, c], path.to_a)

        path = self.graph.backward(c)
        assert_equal([c], path.to_a)

        path = self.graph.backward(b)
        assert_equal([b], path.to_a)

        path = self.graph.backward(a)
        assert_equal([a], path.to_a)
      end

      def test_tensor_num_dimensions
        placeholder = self.graph.placeholder('placeholder_1')
        dims = self.graph.tensor_num_dims(placeholder)
        assert_equal(-1, dims)
      end

      def test_tensor_get_shape
        placeholder = self.graph.placeholder('placeholder_1')
        shape = self.graph.tensor_get_shape(placeholder)
        assert_equal([-1], shape)
      end

      def test_tensor_set_shape
        placeholder = self.graph.placeholder('placeholder_1')
        self.graph.tensor_set_shape(placeholder, [2, -1])
        dims = self.graph.tensor_num_dims(placeholder)
        assert_equal(2, dims)

        shape = self.graph.tensor_get_shape(placeholder)
        assert_equal([2, -1], shape)
      end

      def test_make_graph
        status = Status.new

        placeholder = self.graph.placeholder('placeholder_1')
        attr = placeholder.attr('shape')
        puts attr.value
        constant = self.graph.constant(2, name: 'const_1')
        addn = Math.add_n([placeholder, constant])

        operations = self.graph.operations
        assert_equal(3, operations.count)

        operations = self.graph.operations.to_a
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

        args_0 = self.graph.placeholder("args_0", :int32)
        square1 = Math.square(args_0)

        graph_def = self.graph.export
        puts graph_def.node
      end

      def test_export_import
        placeholder = self.graph.placeholder('feed')
        const = self.graph.constant(3, name: 'scalar')
        op_desc = OperationDescription.new(self.graph, 'Neg', [], name: 'neg')
        op_desc.add_input(const)
        negate = op_desc.save

        assert(self.graph.operation('feed'))
        assert(self.graph.operation('scalar'))
        assert(self.graph.operation('neg'))

        # Get def
        graph_def = self.graph.export

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
