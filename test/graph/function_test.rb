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
        host_graph.copy_function(function)

        op_desc = OperationDescription.new(host_graph, 'MyFunc', [], name: 'MyFunc_0')
        func_operation = op_desc.save

        session = Session.new(host_graph, SessionOptions.new)
        result = session.run({}, [func_operation])
        assert_equal(:int32, result.dtype)
        assert_equal(0, result.shape.length)
        assert_equal(4, result.byte_size)
        assert_equal(10, result.value)

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

        op_desc = OperationDescription.new(func_graph, 'Neg', [], name: 'neg_1')
        op_desc.add_input(feed)
        negate = op_desc.save

        function = func_graph.to_function('MyFunc', nil, [feed], [negate], ['negated_num'])
        assert_equal('MyFunc', function.name)

        host_graph = Graph.new
        host_graph.copy_function(function)

        func_feed = host_graph.placeholder('placeholder_1')

        op_desc = OperationDescription.new(host_graph, 'MyFunc', [], name: 'MyFunc_0')
        op_desc.add_input(func_feed)
        func_op = op_desc.save

        session = Session.new(host_graph, SessionOptions.new)
        result = session.run({func_feed  => Tensor.new(3)}, [func_op])
        assert_equal(:int32, result.dtype)
        assert_equal(0, result.shape.length)
        assert_equal(4, result.byte_size)
        assert_equal(-3, result.value)

        session.close
      end

      def test_oneop_twoinputs_twoduplicateoutputs
        #
        #     |  |
        #     v  v
        #      add
        #       |
        #     +-+-+
        #     |   |
        #     v   v

        func_graph = Graph.new
        feed1 = func_graph.placeholder('feed1')
        feed2 = func_graph.placeholder('feed2')

        op_desc = OperationDescription.new(func_graph, 'AddN', [], name: 'add')
        op_desc.add_input_list([feed1, feed2])
        add = op_desc.save

        function = func_graph.to_function('MyFunc', nil, [feed1, feed2], [add, add], ['output1', 'output2'])
        assert_equal('MyFunc', function.name)

        host_graph = Graph.new
        host_graph.copy_function(function)

        constant = host_graph.constant(2, 'scalar2')
        func_feed = host_graph.placeholder('placeholder_1')

        op_desc = OperationDescription.new(host_graph, 'MyFunc', [], name: 'MyFunc_0')
        op_desc.add_input(constant)
        op_desc.add_input(func_feed)
        func_op = op_desc.save

        session = Session.new(host_graph, SessionOptions.new)
        result = session.run({func_feed => 3}, [func_op])

        assert_equal(:int32, result.dtype)
        assert_equal(0, result.shape.length)
        assert_equal(4, result.byte_size)
        assert_equal(5, result.value)

        session.close
      end
    end
  end
end
