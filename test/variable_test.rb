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
      Tensorflow.disable_eager_execution
      Graph::Graph.new.as_default do |graph|
        var = Variable.new(32)
        assert_kind_of(Graph::Operation, var.handle)
        assert_equal(:int32, var.dtype)

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(var.initializer)
        result = session.run(var.value)
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
      v + 5.0
      assert_equal(8.0, v.value)
    end

    def test_addition_graph
      Graph::Graph.new.as_default do |graph|
        v = Variable.new(3.0)
        operation = v + 5.0

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(v.initializer)
        session.run(operation)
        result = session.run(v.value)
        assert_equal(8, result)
      end
    end

    def test_subtraction
      v = Variable.new(0.0)
      x = v - 1.0
      assert_equal(0.0, v.value)
      assert_equal(-1.0, x.value)
    end

    def test_subtraction
      v = Variable.new(3.0)
      v - 5.0
      assert_equal(-2.0, v.value)
    end

    def test_subtraction_graph
      Graph::Graph.new.as_default do |graph|
        v = Variable.new(3.0)
        operation = v - 5.0

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        session.run(v.initializer)
        session.run(operation)
        result = session.run(v.value)
        assert_equal(-2.0, result)
      end
    end

    def test_global_variables
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([1, 2], name: "v")
        w = Variable.new([3, 4], name: "w")

        global_variables = graph.collection(Graph::GraphKeys::GLOBAL_VARIABLES)
        assert_equal(2, global_variables.length)
        assert_equal(v, global_variables.to_a[0])
        assert_equal(w, global_variables.to_a[1])
      end
    end

    def test_variables_uninitialized
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([1, 2], name: "v")
        w = Variable.new([3, 4], name: "w")

        global_variables = graph.collection(Graph::GraphKeys::GLOBAL_VARIABLES)
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

    def test_shared_name
      Graph::Graph.new.as_default do |graph|
        session = Graph::Session.new(graph, Graph::SessionOptions.new)

        v = Variable.new(300.0, shared_name: 'var4')
        session.run(v.initializer)

        w = Variable.new(nil, shared_name: 'var4', dtype: v.dtype)
        result = session.run(w.value)
        assert_equal(300.0, result)
      end
    end

    def test_variables_initialized
      Graph::Graph.new.as_default do |graph|
        v = Variable.new([1, 2], name: "v")
        w = Variable.new([3, 4], name: "w")

        initializer = Tensorflow.global_variables_initializer

        session = Graph::Session.new(graph, Graph::SessionOptions.new)
        result = session.run(initializer)

        operations = Tensorflow.global_variables.each do |variable|
          assert(variable.initialized?)
        end
      end
    end
  end
end