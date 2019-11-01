require_relative "../test_helper"

module Tensorflow
  module Graph
    class SessionTest < Minitest::Test
      def test_run_scalar
        status = Status.new
        graph = Graph.new

        # Setup graph
        placeholder = graph.placeholder('placeholder', :int32)
        constant = graph.constant(2)
        addn = Math.add_n([placeholder, constant])

        session = Session.new(graph, SessionOptions.new)
        result = session.run([addn], placeholder => Tensor.new(3))
        assert_equal(5, result)

        session.close
      end

      def test_run_array
        status = Status.new
        graph = Graph.new

        # Setup graph
        placeholder = graph.placeholder('placeholder', :int32)
        square = Math.square(placeholder)

        session = Session.new(graph, SessionOptions.new)
        result = session.run([square], placeholder => [[1, 2, 3], [4, 5, 6]])
        assert_equal([[1, 4, 9], [16, 25, 36]], result.to_a)
      end

      def test_test_multiple_outputs
        graph = Graph.new
        w = graph.constant(1.0, shape: [2, 2])
        x = graph.constant(1.0, shape: [2, 2])
        wx = Linalg.matmul(w, x)

        # Split x generates two arrays at outputs (since num_split is set to two)
        split_wx = Tf.split(wx, 0, num_split: 2)

        session = Session.new(graph, SessionOptions.new)
        result = session.run(split_wx)
        session.close

        assert_equal([[[2.0, 2.0]], [[2.0, 2.0]]], result.map(&:to_a))
      end
    end
  end
end