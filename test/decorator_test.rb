require_relative "base_test"

module Tensorflow
  module Graph
    class DecoratorTest < BaseTest
      extend Tensorflow::Decorator

      def setup
        Tensorflow.execution_mode = Tensorflow::GRAPH_MODE
      end

      @tf.function([[:int32]])
      def oneop_oneinput_oneoutput(x)
        Math.negative(x)
      end

      def test_oneop_oneinput_oneoutput
        #
        #           |
        #           v
        #         negate
        #           |
        #           v


        Graph.new.as_default do |host_graph|
          function = oneop_oneinput_oneoutput(nil)
          assert_equal('oneop_oneinput_oneoutput', function.name)

          func_feed = Tensorflow.placeholder(:int32, name: 'placeholder_1')
          func_op = host_graph.create_operation(function.name, [func_feed], name: 'MyFunc_0')

          session = Session.new(host_graph, SessionOptions.new)
          result = session.run([func_op], func_feed  => Tensor.new(3))
          assert_equal(-3, result)

          session.close
        end
      end
    end
  end
end