require_relative "../test_helper"

module Tensorflow
  module Graph
    class OperationDescriptionTest < Minitest::Test
      def graph
        @graph || Graph.new
      end
      
      def setup
        begin
          op_def = self.graph.op_def('CApiAttributesTestOpString')
        rescue TensorflowError
          # Registering placeholder as type causes an exception
          types = FFI::AttrType.symbols.map(&:to_s) - ['placeholder']
          types.each do |name|
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
      end

      def test_set_device
        tensor = Tensor.new(10)
        op_desc = OperationDescription.new(self.graph, 'Const', [], name: 'scalar10')
        op_desc.setup_attr('value', tensor)
        op_desc.setup_attr('dtype', tensor.dtype)
        op_desc.device = "/cpu:0"
        operation = op_desc.save
        assert(operation)
      end

      def test_placeholder
        op_desc = OperationDescription.new(self.graph, 'Placeholder', [], name: 'feed', dtype: :int32, shape: [1,2])
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

      def test_attr_bool
        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpBool', [], name: 'arg1', v: true)
        operation = op_desc.save
        assert(operation.attr('v').bool)
        assert(operation.attr('v').value)

        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpBool', [], name: 'arg1', v: false)
        operation = op_desc.save
        refute(operation.attr('v').bool)
        refute(operation.attr('v').value)
      end

      def test_attr_float
        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpFloat', [], name: 'arg1', v: 77.7)
        operation = op_desc.save

        assert_in_delta(77.7, operation.attr('v').float, 0.00001)
        assert_in_delta(77.7, operation.attr('v').value, 0.00001)
      end

      def test_attr_func
        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpFunc', [], name: 'arg1', v: "my_function_name")
        operation = op_desc.save

        # There doesn't seem to be a way to read back a function attr in the c api.
      end

      def test_attr_int
        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpInt', [], name: 'arg1', v: 77)
        operation = op_desc.save

        assert_equal(77, operation.attr('v').int)
        assert_equal(77, operation.attr('v').value)
      end

      def test_attr_shape
        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpShape', [], name: 'arg1', v: [3,2])
        operation = op_desc.save

        assert_equal([3, 2], operation.attr('v').shape)
        assert_equal([3, 2], operation.attr('v').value)
      end

      def test_attr_string
        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpString', [], name: 'arg1', v: 'Bunny')
        operation = op_desc.save

        assert_equal('Bunny', operation.attr('v').string)
        assert_equal('Bunny', operation.attr('v').value)
      end

      def test_attr_tensor
        tensor = Tensor.new([5, 7])
        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpTensor', [], name: 'arg1', v: tensor)
        operation = op_desc.save

        assert_equal(tensor.value, operation.attr('v').tensor.value)
        assert_equal(tensor.value, operation.attr('v').value.value)
      end

      def test_attr_type
        op_desc = OperationDescription.new(self.graph, 'CApiAttributesTestOpType', [], name: 'arg1', v: :complex64)
        operation = op_desc.save
        assert_equal(:complex64, operation.attr('v').dtype)
        assert_equal(:complex64, operation.attr('v').value)
      end
    end
  end
end