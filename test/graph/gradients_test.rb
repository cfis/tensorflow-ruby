require_relative '../test_helper'

module Tensorflow
  module Graph
    class GradientsTest < Minitest::Test
      def test_derivatives_simple
        graph = Graph.new

        x = graph.constant(3.0, 'x')
        pow = Math.pow(x, 2.0)

        gradients = Gradients.new(graph)
        derivatives = gradients.derivatives(pow, [x])

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, derivatives.flatten)
        assert_equal(6.0, result)

        session.close
      end

      def test_derivatives_chain
        graph = Graph.new
        a = graph.constant(1.0, 'a')
        b = graph.constant(2.1, 'b')
        y = Math.add(Math.pow(a, 2.0), b)
        z = Math.sin(y)

        gradients = Gradients.new(graph)
        derivatives = gradients.derivatives(z, [a, b])

        session = Session.new(graph, SessionOptions.new)
        result = session.run({}, derivatives)
        session.close

        assert_equal(2, result.length)
        assert_in_delta(-1.9983, result[0], 0.1)
        assert_in_delta(-0.9991, result[1], 0.1)
      end
    end
  end
end
