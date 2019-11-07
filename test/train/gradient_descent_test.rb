require_relative "../test_helper"

module Tensorflow
  class GradientDescentTest < Minitest::Test
    def setup
      Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
    end

    def test_basic
      session = Graph::Session.new(Graph::Graph.default, Graph::SessionOptions.new)

      var0 = Variable.new([1.0, 2.0], dtype: :float, name: 'a')
      var1 = Variable.new([3.0, 4.0], dtype: :float, name: 'b')
      loss = var0 * 5.0 + var1 * 3.0

      zeros = Tensorflow.zeros([], dtype: :int64)
      global_step = Variable.new(zeros, name: 'global_step', dtype: :int64)
      optimizer = Train::GradientDescentOptimizer.new(3.0)

      session = Graph::Session.new(Graph::Graph.default, Graph::SessionOptions.new)
      session.run(Tensorflow.global_variables_initializer)

      # Validate initial values
      result = session.run([var0.value, var1.value])
      assert_equal([1.0, 2.0], result[0])
      assert_equal([3.0, 4.0], result[1])

      # Run 1 step of sgd through optimizer
      opt_op = optimizer.minimize(loss, global_step: global_step, var_list: [var0, var1])

      session.run(opt_op)
      result = session.run([var0.value, var1.value])
      assert_equal([-14.0, -13.0], result[0])
      assert_equal([-6.0, -5.0], result[1])
    end
  end
end