require_relative '../test_helper'

module Tensorflow
  module Graph
    class GradientsTest < Minitest::Test
      def setup
        Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
      end

      def test_gradients_simple
        Graph.new.as_default do |graph|
          x = Tensorflow.constant(3.0, name: 'x')
          pow = Math.pow(x, 2.0)

          gradients = Gradients.new(graph)
          gradients = gradients.gradients(pow, [x])

          session = Session.new(graph, SessionOptions.new)
          result = session.run(gradients.flatten)
          assert_equal(6.0, result)

          session.close
        end
      end

      def test_gradients_chain
        Graph.new.as_default do |graph|
          a = Tensorflow.constant(1.0, name: 'a')
          b = Tensorflow.constant(2.1, name: 'b')
          y = Math.add(Math.pow(a, 2.0), b)
          z = Math.sin(y)

          gradients = Gradients.new(graph)
          gradients = gradients.gradients(z, [a, b])

          session = Session.new(graph, SessionOptions.new)
          result = session.run(gradients)
          session.close

          assert_equal(2, result.length)
          assert_in_delta(-1.9983, result[0], 0.1)
          assert_in_delta(-0.9991, result[1], 0.1)
        end
      end

      def test_gradients
        Graph.new.as_default do |graph|
          inp = Tensorflow.constant(1.0, shape: [32, 100], name: "in")
          w = Tensorflow.constant(1.0, shape: [100, 10], name: "w")
          b = Tensorflow.constant(1.0, shape: [10], name: "b")
          xw = Linalg.matmul(inp, w)#, name: "xw")
          h = NN.bias_add(xw, b)#, name="h")

          gradients = Gradients.new(graph)
          w_grad = gradients.gradients(h, [w])

          session = Session.new(graph, SessionOptions.new)
          result = session.run(w_grad)
          session.close

          expected = Numo::Float32.new([100, 10]).fill(32)
          assert_equal(expected.to_a, result)
        end
      end

      def test_unused_ouput
        Graph.new.as_default do |graph|
          w = Tensorflow.constant(1.0, shape: [2, 2])
          x = Tensorflow.constant(1.0, shape: [2, 2])

          wx = Linalg.matmul(w, x)
          split_wx = Tensorflow.split(wx, 0, num_split: 2)
          c = Math.reduce_sum(split_wx)

          gradients = Gradients.new(graph)
          gw = gradients.gradients(c, [w])


          session = Session.new(graph, SessionOptions.new)
          result = session.run([gw])
          session.close

          assert_equal([[2.0, 2.0], [2.0, 2.0]], result)
        end
      end
    end
  end
end
