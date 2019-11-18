require_relative "../test_helper"

module Tensorflow
  module Graph
    class FunctionDefTest < Minitest::Test
      extend Decorator

      def setup
        Graph.reset_default
      end

      @tf.function
      def simple
        Tensorflow.constant(32)
      end

      def test_simple_eager
        Eager::Context.default.as_default do |context|
          operation = context.create_operation(self.simple)
          result = context.execute(operation)
          assert_equal(32, result.value)
        end
      end

      def test_simple_graph
        Graph.default.as_default do |graph|
          operation = graph.create_operation(self.simple)
          result = Session.run(Graph.default) do |session|
                     session.run(operation)
                   end
          assert_equal(32, result)
        end
      end

      @tf.function(input_signatures=[[:int32]])
      def one_parameter(value)
        value * 5
      end

      def test_one_parameter_eager
        Eager::Context.default.as_default do |context|
          operation = context.create_operation(self.one_parameter, [4])
          result = context.execute(operation)
          assert_equal(20, result.value)
        end
      end

      def test_one_parameter_graph
        Graph.default.as_default do |graph|
          operation = graph.create_operation(self.one_parameter, [4])
          result = Session.run(Graph.default) do |session|
            session.run(operation)
          end
          assert_equal(20, result)
        end
      end

      @tf.function(input_signatures=[[:int32], [:int32]])
      def two_parameter(x, y)
        x * y
      end

      def test_two_parameters_eager
        Eager::Context.default.as_default do |context|
          operation = context.create_operation(self.two_parameter, [7, 11])
          result = context.execute(operation)
          assert_equal(77, result.value)
        end
      end

      def test_two_parameters_graph
        Graph.default.as_default do |graph|
          operation = graph.create_operation(self.two_parameter, [7, 11])
          result = Session.run(Graph.default) do |session|
            session.run(operation)
          end
          assert_equal(77, result)
        end
      end
    end
  end
end
