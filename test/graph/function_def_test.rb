require_relative "../test_helper"

module Tensorflow
  module Graph
    class FunctionDefTest < Minitest::Test
      extend Decorator

      def setup
        Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
        Graph.reset_default
      end

      @tf.function
      def simple
        Tensorflow.constant(32)
      end

      def test_simple
        operation = self.simple
        result = Session.run(Graph.default) do |session|
                   session.run(operation)
                 end
        assert_equal(32, result)
      end

      @tf.function(input_signatures=[[:int32]])
      def one_parameter(value)
        value * 5
      end

      def test_one_parameter
        operation = self.one_parameter(4)
        result = Session.run(Graph.default) do |session|
          session.run(operation)
        end
        assert_equal(20, result)
      end

      @tf.function(input_signatures=[[:int32], [:int32]])
      def two_parameter(x, y)
        x * y
      end

      def test_two_parameters
        operation = self.two_parameter(7, 11)
        result = Session.run(Graph.default) do |session|
          session.run(operation)
        end
        assert_equal(77, result)
      end
    end
  end
end
