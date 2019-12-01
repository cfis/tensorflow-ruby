require_relative "../base_test"

module Tensorflow
  class GradientDescentTest < BaseTest
    def setup
      Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
    end

    def test_basic
      [:float, :double].each do |dtype|
        Graph::Graph.reset_default
        var0 = Variable.new([1.0, 2.0], dtype: dtype, name: 'a')
        var1 = Variable.new([3.0, 4.0], dtype: dtype, name: 'b')
        loss = var0 * 5.0 + var1 * 3.0

        zeros = Tensorflow.zeros([], dtype: :int64)
        global_step = Variable.new(zeros, name: 'global_step', dtype: :int64)
        optimizer = Train::GradientDescentOptimizer.new(3.0)

        session = Graph::Session.new(Graph::Graph.default, Graph::SessionOptions.new)
        session.run(Tensorflow.global_variables_initializer)

        # Validate initial values
        result = session.run([var0, var1])
        assert_equal([1.0, 2.0], result[0])
        assert_equal([3.0, 4.0], result[1])

        # Run 1 step of sgd through optimizer
        opt_op = optimizer.minimize(loss, global_step: global_step, var_list: [var0, var1])

        session.run(opt_op)
        result = session.run([var0, var1])
        assert_equal([-14.0, -13.0], result[0])
        assert_equal([-6.0, -5.0], result[1])
      end
    end

    def test_no_variables
      Graph::Graph.reset_default

      optimizer = Train::GradientDescentOptimizer.new(3.0)

      loss = Tensorflow.constant(5.0)
      exception = assert_raises(Error::InvalidArgumentError) do
        optimizer.minimize(loss)
      end
      assert_equal('There are no variables to train for the loss function', exception.message)
    end

    def test_no_gradient
      Graph::Graph.reset_default
      var0 = Variable.new([1.0, 2.0], dtype: :float, name: 'a')
      var1 = Variable.new([3.0, 4.0], dtype: :float, name: 'b')
      loss = var0 * 5.0

      optimizer = Train::GradientDescentOptimizer.new(3.0)
      exception = assert_raises(Error::InvalidArgumentError) do
        optimizer.minimize(loss, var_list: [var1])
      end
      assert_equal('No gradients provided for any variable, check your graph for ops that do not support gradients', exception.message)
    end

    def test_callable_learning_rate
      Graph::Graph.reset_default
      var0 = Variable.new([1.0, 2.0], dtype: :float, name: 'a')
      var1 = Variable.new([3.0, 4.0], dtype: :float, name: 'b')
      loss = var0 * 5.0 + var1 * 3.0

      zeros = Tensorflow.zeros([], dtype: :int64)
      global_step = Variable.new(zeros, name: 'global_step', dtype: :int64)

      learning_rate = -> {4.0}
      optimizer = Train::GradientDescentOptimizer.new(3.0)

      session = Graph::Session.new(Graph::Graph.default, Graph::SessionOptions.new)
      session.run(Tensorflow.global_variables_initializer)

      # Run 1 step of sgd through optimizer
      opt_op = optimizer.minimize(loss, global_step: global_step, var_list: [var0, var1])

      session.run(opt_op)
      result = session.run([var0])
      assert_equal([-14.0, -13.0], result)

      result = session.run([var1])
      assert_equal([-6.0, -5.0], result)
    end
  end
end