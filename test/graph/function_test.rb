require_relative "../test_helper"

module Tensorflow
  module Graph
    class FuntionTest < Minitest::Test
      def test_oneop_zeroinputs_oneoutput
        #
        #      constant
        #         |
        #         v

        function = Graph.new.as_default do |func_graph|
          const = Tensorflow.constant(10, name: 'scalar10')
          func_graph.to_function('MyFunc', nil, nil, [const], ['output1'])
        end
        assert_equal('MyFunc', function.name)

        Graph.new.as_default do |host_graph|
          host_graph.copy_function(function)

          op_desc = OperationDescription.new(host_graph, 'MyFunc', [], name: 'MyFunc_0')
          func_operation = op_desc.save

          session = Session.new(host_graph, SessionOptions.new)
          result = session.run([func_operation])
          assert_equal(10, result)

          session.close
        end
      end

      def test_oneop_oneinput_oneoutput
        #
        #           |
        #           v
        #         negate
        #           |
        #           v

        function = Graph.new.as_default do |func_graph|
          feed = Tensorflow.placeholder('placeholder_1')
          negate = Math.negative(feed)
          func_graph.to_function('MyFunc', nil, [feed], [negate], ['negated_num'])
        end
        assert_equal('MyFunc', function.name)

        Graph.new.as_default do |host_graph|
          host_graph.copy_function(function)

          func_feed = Tensorflow.placeholder('placeholder_1')

          op_desc = OperationDescription.new(host_graph, 'MyFunc', [], name: 'MyFunc_0')
          op_desc.add_input(func_feed)
          func_op = op_desc.save

          session = Session.new(host_graph, SessionOptions.new)
          result = session.run([func_op], func_feed  => Tensor.new(3))
          assert_equal(-3, result)

          session.close
        end
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

        function = Graph.new.as_default do |func_graph|
          feed1 = Tensorflow.placeholder('feed1')
          feed2 = Tensorflow.placeholder('feed2')

          op_desc = OperationDescription.new(func_graph, 'AddN', [], name: 'add')
          op_desc.add_input_list([feed1, feed2])
          add = op_desc.save

          func_graph.to_function('MyFunc', nil, [feed1, feed2], [add, add], ['output1', 'output2'])
        end
        assert_equal('MyFunc', function.name)

        Graph.new.as_default do |host_graph|
          host_graph.copy_function(function)

          constant = Tensorflow.constant(2, name: 'scalar2')
          func_feed = Tensorflow.placeholder('placeholder_1')

          op_desc = OperationDescription.new(host_graph, 'MyFunc', [], name: 'MyFunc_0')
          op_desc.add_input(constant)
          op_desc.add_input(func_feed)
          func_op = op_desc.save

          session = Session.new(host_graph, SessionOptions.new)
          result = session.run([func_op], func_feed => 3)
          assert_equal([5, 5], result)

          session.close
        end
      end
    end
  end
end
