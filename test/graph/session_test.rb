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
        result = session.run({placeholder => Tensor.new(3)}, [addn])
        assert_equal(:int32, result.dtype)
        assert_equal(0, result.shape.length)
        assert_equal(4, result.byte_size)
        assert_equal(5, result.value)

        session.close
      end

      def test_run_array
        status = Status.new
        graph = Graph.new

        # Setup graph
        placeholder = graph.placeholder('placeholder', :int32)
        square = Math.square(placeholder)

        session = Session.new(graph, SessionOptions.new)
        result = session.run({placeholder => [[1, 2, 3], [4, 5, 6]]}, [square])
        assert_equal(:int32, result.dtype)
        assert_equal(2, result.shape.length)
        assert_equal(24, result.byte_size)
        assert_equal([[1, 4, 9], [16, 25, 36]], result.value)
      end
    end
  end
end