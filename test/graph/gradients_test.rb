require_relative '../test_helper'

module Tensorflow
  module Graph
    class GradientsTest < Minitest::Test
      # def test_operations
      #   graph = Graph.new
      #
      #   x = graph.constant(3.0, name='x')
      #   two = graph.constant(2.0, name='x')
      #
      #   y = graph.create_operation('Pow') do |op_desc|
      #     op_desc.add_input(x)
      #     op_desc.add_input(two)
      #   end
      #
      #   gradients = Gradients.new(graph)
      #   operations = gradients.find_operations(y, x)
      #   assert_equal(2, operations.length)
      #
      #   operation = operations[0]
      #   assert_equal(x, operation)
      #
      #   operation = operations[1]
      #   assert_equal(y, operation)
      # end

      def test_derivate_ops
        graph = Graph.new

        x = graph.constant(3.0, 'x')
        two = graph.constant(2.0, 'two')

        y = graph.create_operation('Pow', 'pow') do |op_desc|
          op_desc.add_input(x)
          op_desc.add_input(two)
        end

        gradients = Gradients.new(graph)
        foo = gradients.derivative(y, x)
        #assert_equal(2, operations.length)
        a = foo
      end
    end
  end
end
