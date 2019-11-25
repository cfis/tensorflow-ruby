require_relative "../test_helper"

module Tensorflow
  module Graph
    class OperationDescriptionTest < Minitest::Test
      def graph
        @graph ||= Graph.new
      end
      
      def setup
        begin
          op_def = self.graph.op_def('CApiAttributesTestOpString')
        rescue Error::InvalidArgumentError
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

      def test_addn
        Graph.new.as_default do |graph|
          constant1 = Tensorflow.constant(12)
          constant2 = Tensorflow.constant(15)
          add = Math.add_n([constant1, constant2])

          session = Session.new(graph, SessionOptions.new)
          result = session.run(add)
          assert_equal(27, result)
        end
      end

      def test_pack
        self.graph.as_default do
          data = Numo::NArray[[2,2], [2,2]]
          split = Tensorflow.split(data, 0, num_split: 2)
          sum = Math.reduce_sum(split)

          session = Session.new(self.graph, SessionOptions.new)
          result = session.run([sum])
          session.close

          assert_equal(8, result)
        end
      end

      def test_single_output
        self.graph.as_default do
          data = Numo::NArray[[2,2], [2,2]]
          split = Tensorflow.split(data, 0, num_split: 2)
          sum = Math.reduce_sum(split[1])

          session = Session.new(self.graph, SessionOptions.new)
          result = session.run([sum])
          session.close

          assert_equal(4, result)
        end
      end

      def test_capture
        # Create a constant in graph 1
        constant = nil
        g1 = Graph.new.as_default do |graph|
               constant = Tensorflow.constant(12, name: "CaptureMe")
             end

        # Create graph 2 and negate the value
        Graph.new.as_default do |graph|
          neg = Math.negative(constant)

          session = Session.new(graph, SessionOptions.new)
          result = session.run(neg)
          assert_equal(-12, result)
        end
      end

      def test_capture_invalid
        # Create a constant in graph 1
        placeholder = nil
        g1 = Graph.new.as_default do |graph|
          placeholder = Tensorflow.placeholder(:int32, name: "CaptureMe")
        end

        # Create graph 2 and negate the value
        Graph.new.as_default do |graph|
          exception = assert_raises(Error::InvalidArgumentError) do
            Math.negative(placeholder)
          end
          assert_equal("Cannot capture a placeholder by value (name: CaptureMe, type: Placeholder)", exception.message)
        end
      end
    end
  end
end