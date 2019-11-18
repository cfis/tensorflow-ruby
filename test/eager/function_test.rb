require_relative "../test_helper"

module Tensorflow
  module Eager
    class FunctionTest < Minitest::Test
      def setup
        Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
      end

      def create_function
        Graph::Graph.new.as_default do |func_graph|
          const = Tensorflow.constant(10, name: 'scalar10')
          func_graph.to_function('MyFunc', nil, nil, [const], ['output1'])
        end
      end

      def test_function
        context = Eager::Context.new

        function = create_function
        context.add_function(function)
        assert(context.function?(function))

        context.remove_function(function)
        refute(context.function?(function))
      end

      def test_function_by_name
        context = Eager::Context.new
        refute(context.function?('MyFunc'))

        function = create_function
        context.add_function(function)
        assert(context.function?('MyFunc'))

        context.remove_function(function)
        refute(context.function?('MyFunc'))
      end

      def test_execute_function
        context = Eager::Context.new
        function = create_function
        context.add_function(function)

        operation = context.create_operation(function)
        result = context.execute(operation)
        assert_equal(10, result.value)
      end
    end
  end
end
