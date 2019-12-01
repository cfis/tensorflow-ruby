require_relative "../base_test"

module Tensorflow
  module Graph
    class GraphTest < BaseTest
      def setup
        Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
      end

      def test_create
        assert(Graph.new)
      end

      def test_as_default
        graph = Graph.new
        graph.as_default do |g|
          assert_same(graph, g)
          refute_same(Graph.default, g)
        end
      end

      def test_as_default_const
        const_1 = Tensorflow.constant('const1')
        assert_same(Graph.default, const_1.graph)
        Graph.new.as_default do |graph|
          const_2 = Tensorflow.constant('const2')
          assert_same(graph, const_2.graph)
        end
      end

      def test_op_def
        op_def = Graph.default.op_def('Variable')
        assert(op_def)
        assert_equal('Variable', op_def.name)
      end

      def test_operations
        graph = Graph.new
        assert_empty(graph.operations.to_a)
      end

      def test_forward
        a = Tensorflow.constant(1.0, name: 'a')
        b = Tensorflow.constant(2.1, name: 'b')
        c = Tensorflow.constant(2.0, name: 'c')
        d = Math.pow(a, c)
        y = Math.add(d, b)
        z = Math.sin(y)

        path = Graph.default.forward(a)
        assert_equal([a, d, y, z], path.to_a)

        path = Graph.default.forward(b)
        assert_equal([b, y, z], path.to_a)

        path = Graph.default.forward(c)
        assert_equal([c, d, y, z], path.to_a)

        path = Graph.default.forward(d)
        assert_equal([d, y, z], path.to_a)

        path = Graph.default.forward(y)
        assert_equal([y, z], path.to_a)

        path = Graph.default.forward(z)
        assert_equal([z], path.to_a)
      end

      def test_backward
        a = Tensorflow.constant(1.0, name: 'a')
        b = Tensorflow.constant(2.1, name: 'b')
        c = Tensorflow.constant(2.0, name: 'c')
        d = Math.pow(a, c)
        y = Math.add(d, b)
        z = Math.sin(y)

        path = Graph.default.backward(z)
        assert_equal([z, y, d, a, c, b], path.to_a)

        path = Graph.default.backward(y)
        assert_equal([y, d, a, c, b], path.to_a)

        path = Graph.default.backward(d)
        assert_equal([d, a, c], path.to_a)

        path = Graph.default.backward(c)
        assert_equal([c], path.to_a)

        path = Graph.default.backward(b)
        assert_equal([b], path.to_a)

        path = Graph.default.backward(a)
        assert_equal([a], path.to_a)
      end

      def test_tensor_num_dimensions
        placeholder = Tensorflow.placeholder(:int32, name: 'placeholder_1')
        dims = Graph.default.output_shapes(placeholder)
        assert_equal([[]], dims)
      end

      def test_tensor_get_shape
        placeholder = Tensorflow.placeholder(:int32, name: 'placeholder_1')
        shape = Graph.default.output_shapes(placeholder)
        assert_equal([[]], shape)
      end

      def test_tensor_set_shape
        placeholder = Tensorflow.placeholder(:int32, name: 'placeholder_1')
        Graph.default.tensor_set_shape(placeholder, [2, -1])
        shapes = Graph.default.output_shapes(placeholder)
        assert_equal([[2, -1]], shapes)
      end

      def test_make_graph
        status = Status.new

        Graph.new.as_default do |graph|
          placeholder = Tensorflow.placeholder(:int32, name: 'placeholder_1')
          constant = Tensorflow.constant(2, name: 'const_1')
          addn = Math.add_n([placeholder, constant])
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
      end

      def test_control_inputs
        Graph.new.as_default do |graph|
          feed1 = Tensorflow.placeholder(:int32, name: 'feed1')

          feed2 = Tensorflow.placeholder(:int32, name: 'feed2')

          constant = Tensorflow.constant(5, name: 'scalar5')

          add = graph.control_dependencies([constant]) do
            Math.add(feed1, feed2)
          end

          assert_equal(0, feed1.num_control_inputs)
          assert_equal(0, feed1.num_control_outputs)

          assert_equal(0, feed2.num_control_inputs)
          assert_equal(0, feed2.num_control_outputs)

          assert_equal(0, constant.num_control_inputs)
          assert_equal(1, constant.num_control_outputs)
          assert_equal(add, constant.control_outputs[0])

          assert_equal(1, add.num_control_inputs)
          assert_equal(1, add.control_inputs.length)
          assert_equal(constant, add.control_inputs[0])
          assert_equal(0, add.num_control_outputs)
        end
      end

      def test_graph_def
        args_0 = Tensorflow.placeholder(:int32, name: "args_0")
        square1 = Math.square(args_0)

        graph_def = Graph.default.as_graph_def
        refute_nil(graph_def)
      end

      def test_export_import
        graph_def = nil
        Graph.new.as_default do |graph|
          placeholder = Tensorflow.placeholder(:int32, name: 'feed')
          const = Tensorflow.constant(3, name: 'scalar')
          result = Math.negative(const)

          assert(graph.operation('feed'))
          assert(graph.operation('scalar'))
          assert(graph.operation('Neg'))

          # Get def
          graph_def = graph.as_graph_def
        end

        Graph.new.as_default do |new_graph|
          options = GraphDefOptions.new
          options.prefix = 'imported'
          new_graph.import(graph_def, options)

          assert(new_graph.operation('imported/feed'))
          assert(new_graph.operation('imported/scalar'))
          assert(new_graph.operation('imported/Neg'))
        end
      end
    end
  end
end
