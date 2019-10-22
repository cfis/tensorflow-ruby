require_relative "../test_helper"

module Tensorflow
  module Graph
    class OperationDescriptionTest < Minitest::Test
      def register_ops
        ['string', 'int', 'float', 'bool', 'type', 'shape', 'tensor'].each do |name|
          OpDefBuilder.new("CApiAttributesTestOp#{name.capitalize}")
                      .attr("v: #{name}")
                      .shape_inference(OpDefBuilder.unknown_shape_inference_func)
                      .register

          OpDefBuilder.new("CApiAttributesTestOpList#{name.capitalize}")
              .attr("v: list(#{name})")
              .shape_inference(OpDefBuilder.unknown_shape_inference_func)
              .register
        end
      end

      def test_placeholder
        graph = Graph.new
        op_desc = OperationDescription.new(graph, 'Placeholder', 'feed')
        op_desc.attr('dtype').dtype = :int32
        op_desc.attr('shape').shape = [1,2]
        operation = op_desc.save

        assert_equal('feed', operation.name)
        assert_equal('Placeholder', operation.op_type)
        assert_empty(operation.device)

        attr = operation.attr('dtype')
        assert(attr)
        assert_equal(:int32, attr.dtype)

        attr = operation.attr('shape')
        assert_equal([1, 2], attr.shape)
        assert(attr)
      end

      def test_tensor_attr
        self.register_ops
        graph = Graph.new
        tensor = Tensor.new([5, 7])

        op_desc = OperationDescription.new(graph, 'CApiAttributesTestOpTensor', 'arg1')
        op_desc.attr('v').tensor = tensor
        operation = op_desc.save

        new_tensor = operation.attr('v').tensor
        assert_equal(tensor.value, new_tensor.value)
      end

      def test_set_device
        graph = Graph.new
        tensor = Tensor.new(10)
        op_desc = OperationDescription.new(graph, 'Const', 'scalar10')
        op_desc.attr('value').tensor = tensor
        op_desc.attr('dtype').dtype = tensor.dtype
        op_desc.device = "/cpu:0"
        operation = op_desc.save
        assert(operation)
      end
    end
  end
end