require_relative "../test_helper"

module Tensorflow
  module Graph
    class FuntionTest < Minitest::Test
      def test_oneop_zeroinputs_oneoutput
        #
        #      constant
        #         |
        #         v

        func_graph = Graph.new
        const = func_graph.constant(10, 'scalar10')

        function = func_graph.to_function('MyFunc', nil, nil, [const], ['output1'])
        assert_equal('MyFunc', function.name)

        host_graph = Graph.new
        function.copy_to(host_graph)

        op_desc = OperationDescription.new(host_graph, 'MyFunc', 'MyFunc_0')
        func_operation = op_desc.save

        session = Session.new(host_graph, SessionOptions.new)
        result = session.run([], [func_operation])
        assert_equal(1, result.length)

        tensor = result[0]
        assert_equal(:int32, tensor.dtype)
        assert_equal(0, tensor.shape.length)
        assert_equal(4, tensor.byte_size)
        assert_equal(10, tensor.value)

        session.close
      end

      def test_oneop_oneinput_oneoutput
        #
        #           |
        #           v
        #         negate
        #           |
        #           v

        func_graph = Graph.new

        feed = func_graph.placeholder('placeholder_1')

        op_desc = OperationDescription.new(func_graph, 'Neg', 'neg_1')
        op_desc.add_input(feed)
        negate = op_desc.save

        function = func_graph.to_function('MyFunc', nil, [feed], [negate], ['negated_num'])
        assert_equal('MyFunc', function.name)

        host_graph = Graph.new
        function.copy_to(host_graph)

        func_feed = host_graph.placeholder('placeholder_1')

        op_desc = OperationDescription.new(host_graph, 'MyFunc', 'MyFunc_0')
        op_desc.add_input(func_feed)
        func_op = op_desc.save

        session = Session.new(host_graph, SessionOptions.new)
        result = session.run([[func_feed, Tensor.new(3)]], [func_op])
        assert_equal(1, result.length)

        tensor = result[0]
        assert_equal(:int32, tensor.dtype)
        assert_equal(0, tensor.shape.length)
        assert_equal(4, tensor.byte_size)
        assert_equal(-3, tensor.value)

        session.close
      end
    end
  end
end
