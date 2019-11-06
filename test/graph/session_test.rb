require_relative "../test_helper"

module Tensorflow
  module Graph
    class SessionTest < Minitest::Test
      def setup
        Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
      end
      
      def test_run_scalar
        Graph.new.as_default do |graph|
          placeholder = Tensorflow.placeholder('placeholder', dtype: :int32)
          constant = Tensorflow.constant(2)
          addn = Math.add_n([placeholder, constant])

          session = Session.new(graph, SessionOptions.new)
          result = session.run([addn], placeholder => Tensor.new(3))
          assert_equal(5, result)

          session.close
        end
      end

      def test_run_array
        Graph.new.as_default do |graph|
          placeholder = Tensorflow.placeholder('placeholder', dtype: :int32)
          square = Math.square(placeholder)

          session = Session.new(graph, SessionOptions.new)
          result = session.run([square], placeholder => [[1, 2, 3], [4, 5, 6]])
          assert_equal([[1, 4, 9], [16, 25, 36]], result.to_a)
        end
      end

      def test_test_multiple_outputs
        Graph.new.as_default do |graph|
          w = Tensorflow.constant(1.0, shape: [2, 2])
          x = Tensorflow.constant(1.0, shape: [2, 2])
          wx = Linalg.matmul(w, x)

          # Split x generates two arrays at outputs (since num_split is set to two)
          split_wx = Tensorflow.split(wx, 0, num_split: 2)

          session = Session.new(graph, SessionOptions.new)
          result = session.run(split_wx)
          session.close

          assert_equal([[[2.0, 2.0]], [[2.0, 2.0]]], result.map(&:to_a))
        end
      end
    end
  end
end