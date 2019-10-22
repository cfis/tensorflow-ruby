require_relative "../test_helper"

module Tensorflow
  module Graph
    class SessionTest < Minitest::Test
      def test_run_graph
        status = Status.new
        graph = Graph.new

        # Add placeholder
        op_desc = OperationDescription.new(graph, 'Placeholder', 'placeholder')
        op_desc.attr('dtype').dtype = :int32
        placeholder = op_desc.save

        # Add constant
        tensor = Tensor.new(2)
        op_desc = OperationDescription.new(graph, 'Const', 'const')
        op_desc.attr('dtype').dtype = tensor.dtype
        op_desc.attr('value').tensor = tensor
        constant = op_desc.save

        # Add add operation
        op_desc = OperationDescription.new(graph, "AddN", 'addn')
        op_desc.add_inputs(placeholder, constant)
        addn = op_desc.save

        session = Session.new(graph, SessionOptions.new)

        result = session.run([[placeholder, Tensor.new(3)]], [addn])
        assert_equal(1, result.length)

        tensor = result[0]
        assert_equal(:int32, tensor.dtype)
        assert_equal(0, tensor.shape.length)
        assert_equal(4, tensor.byte_size)
        assert_equal(5, tensor.value)

        session.close
      end
    end
  end
end