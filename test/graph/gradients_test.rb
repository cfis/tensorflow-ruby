require_relative '../test_helper'

module Tensorflow
  module Graph
    class GradientsTest < Minitest::Test
      # def test_derivatives_simple
      #   graph = Graph.new
      #
      #   x = graph.constant(3.0, name: 'x')
      #   pow = Math.pow(x, 2.0)
      #
      #   gradients = Gradients.new(graph)
      #   derivatives = gradients.derivatives(pow, [x])
      #
      #   session = Session.new(graph, SessionOptions.new)
      #   result = session.run(derivatives.flatten)
      #   assert_equal(6.0, result)
      #
      #   session.close
      # end
      #
      # def test_derivatives_chain
      #   graph = Graph.new
      #   a = graph.constant(1.0, name: 'a')
      #   b = graph.constant(2.1, name: 'b')
      #   y = Math.add(Math.pow(a, 2.0), b)
      #   z = Math.sin(y)
      #
      #   gradients = Gradients.new(graph)
      #   derivatives = gradients.derivatives(z, [a, b])
      #
      #   session = Session.new(graph, SessionOptions.new)
      #   result = session.run(derivatives)
      #   session.close
      #
      #   assert_equal(2, result.length)
      #   assert_in_delta(-1.9983, result[0], 0.1)
      #   assert_in_delta(-0.9991, result[1], 0.1)
      # end
      #
      # def test_gradients
      #   graph = Graph.new
      #   inp = graph.constant(1.0, shape: [32, 100], name: "in")
      #   w = graph.constant(1.0, shape: [100, 10], name: "w")
      #   b = graph.constant(1.0, shape: [10], name: "b")
      #   xw = Linalg.matmul(inp, w)#, name: "xw")
      #   h = NN.bias_add(xw, b)#, name="h")
      #
      #   gradients = Gradients.new(graph)
      #   w_grad = gradients.derivatives(h, [w])
      #
      #   session = Session.new(graph, SessionOptions.new)
      #   result = session.run(w_grad)
      #   session.close
      #
      #   expected = Numo::Float32.new([100, 10]).fill(32)
      #   assert_equal(expected.to_a, result)
      # end

      def test_unused_ouput
        graph = Graph.new
        w = graph.constant(1.0, shape: [2, 2])
        x = graph.constant(1.0, shape: [2, 2])

       # w = Tensor.new(1.0, shape: [2, 2])
        #x = Tensor.new(1.0, shape: [2, 2])

        wx = Linalg.matmul(w, x)
        split_wx = Tf.split(wx, 0, num_split: 2)
        c = Math.reduce_sum(split_wx)

        gradients = Gradients.new(graph)
        gw = gradients.derivatives(c, [w])

        session = Session.new(graph, SessionOptions.new)
        result = session.run([c])
        session.close

        puts result

        assert_equal([[2.0, 2.0], [2.0, 2.0]], result)
      end
    end
  end
end
