require_relative "../test_helper"

module Tensorflow
  module Eager
    class FunctionTest < Minitest::Test
      def create_function
        func_graph = Graph::Graph.new
        const = func_graph.constant(10, name: 'scalar10')
        func_graph.to_function('MyFunc', nil, nil, [const], ['output1'])
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

        operation = context.create_operation(function.name)
        result = context.execute(operation)
        assert_equal(10, result.value)
      end
    end
  end
end
