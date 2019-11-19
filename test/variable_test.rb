require_relative "test_helper"

module Tensorflow
  class VariableTest < Minitest::Test
    def setup
      Tensorflow.execution_mode = Tensorflow::EAGER_MODE
    end

    def test_simple_eager
      var = Variable.new(32)
      assert_kind_of(Eager::TensorHandle, var.handle)
      assert_equal(:int32, var.dtype)
      assert_equal([], var.shape)
      assert_equal(32, var.value)
    end

    def test_simple_graph
      Graph::Graph.new.as_default do |graph|
        var = Variable.new(32)
        assert_kind_of(Graph::Operation, var.handle)
        assert_equal(:int32, var.dtype)

        # Execute the value
        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(var.initializer)
        result = session.run(var.value)
        assert_equal(32, result)

        # Let's check executing the variable does the same thing
        result = session.run(var)
        assert_equal(32, result)
      end
    end

    def test_float
      x = Tensorflow::Variable.new(1.0)
      assert_equal(1.0, x.value)

      handle = Eager::TensorHandle.from_value(ExecutionContext.current, x)
      assert_equal(1.0, x.value)
    end

    def test_value
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      assert_equal([[[0, 1, 2],
                     [3, 4, 5]],
                    [[6, 7, 8],
                     [9, 10, 11]]], var1.value)
    end

    def test_tensor
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      tensor = var1.tensor
      assert_equal([[[0, 1, 2],
                     [3, 4, 5]],
                    [[6, 7, 8],
                     [9, 10, 11]]], tensor.value)
    end

    def test_rank
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      assert_equal(3, var1.rank)
    end

    def test_dtype
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      assert_equal(:int32, var1.dtype)
    end

    def test_shape
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])
      assert_equal([2, 2, 3], var1.shape)
    end

    def test_shape_graph
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([[[0, 1, 2],
                          [3, 4, 5]],
                         [[6, 7, 8],
                          [9, 10, 11]]])

        assert_equal([], v.shape)

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(v.initializer)

        assert_equal([], v.shape)
      end
    end

    def test_reshape
      var1 = Variable.new([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])

      tensor = var1.reshape([2, 6])
      assert_equal(:int32, tensor.dtype)
      assert_equal([2, 6], tensor.shape)
      assert_equal([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], tensor.value)
    end

    def test_assign
      v = Variable.new(3.0)
      v.value = 5.0
      assert_equal(5.0, v.value)
    end

    def test_assign_graph
      Graph::Graph.new.as_default do |graph|
        v = Variable.new(3.0)
        v.value = 5.0

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(v.initializer)
        result = session.run(v.value)
        assert_equal(5.0, result)
      end
    end

    def test_addition
      v = Variable.new(3.0)
      add = v + 5.0
      assert_equal(8.0, add.value)
    end

    def test_addition_graph
      Graph::Graph.new.as_default do |graph|
        v = Variable.new(3.0)
        operation = v + 5.0

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(v.initializer)
        session.run(operation)
        result = session.run(operation)
        assert_equal(8, result)
      end
    end

    def test_subtraction
      v = Variable.new(3.0)
      sub = v - 5.0
      assert_equal(-2.0, sub.value)
    end

    def test_subtraction_graph
      Graph::Graph.new.as_default do |graph|
        v = Variable.new(3.0)
        operation = v - 5.0

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(v.initializer)
        session.run(operation)
        result = session.run(operation)
        assert_equal(-2.0, result)
      end
    end

    def test_multiplication
      v = Variable.new(3.0)
      mul = v * 5.0
      assert_equal(15.0, mul.value)
    end

    def test_multiplication_graph
      Graph::Graph.new.as_default do |graph|
        v = Variable.new(3.0)
        operation = v * 5.0

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(v.initializer)
        session.run(operation)
        result = session.run(operation)
        assert_equal(15, result)
      end
    end

    def test_global_variables
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([1, 2], name: "v")
        w = Variable.new([3, 4], name: "w")

        global_variables = graph.get_collection_ref(Graph::GraphKeys::GLOBAL_VARIABLES)
        assert_equal(2, global_variables.length)
        assert_equal(v, global_variables.to_a[0])
        assert_equal(w, global_variables.to_a[1])
      end
    end

    def test_variables_uninitialized
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([1, 2], name: "v")
        w = Variable.new([3, 4], name: "w")

        global_variables = graph.get_collection_ref(Graph::GraphKeys::GLOBAL_VARIABLES)
        operations = global_variables.map do |variable|
          variable.initialized?
        end

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        result = session.run(operations)
        assert_equal(0, result[0])
        assert_equal(0, result[1])
      end
    end

    def test_read_uninitialized
      Graph::Graph.new.as_default do |graph|
        v = Variable.new(10)

        session = Graph::Session.new(graph, Graph::SessionOptions.new)

        exception = assert_raises(TensorflowError) do
          result = session.run(v.value)
        end
        assert_match(/This could mean that the variable was uninitialized/, exception.to_s)
      end
    end

    def test_variables_initialized
      Graph::Graph.new.as_default do |graph|
        v = Variable.new(1.0, name: "var0")
        w = Variable.new([3, 4], name: "w")

        initializer = Tensorflow.global_variables_initializer

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(initializer)

        operations = Tensorflow.global_variables.map do |variable|
          variable.initialized?
        end

        result = session.run(operations)
        assert_equal(1, result[0])
        assert_equal(1, result[1])
      end
    end

    def test_operation_shape
      shape = [100, 200]
      normal = Random.normal(shape)
      variable = Variable.new(normal)
      assert_equal(shape, variable.shape)
    end

    def test_operation_tensor
      shape = [100, 200]
      data = Numo::Int32.new(20_000).seq.reshape(*shape)
      variable = Variable.new(data)
      assert_equal(shape, variable.shape)
    end
  end
end