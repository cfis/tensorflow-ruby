module Tensorflow
  module Eager
    class TensorHandle
      include TensorMixin
      include Operators

      attr_reader :context

      def self.finalize(pointer)
        proc do
          FFI.TFE_DeleteTensorHandle(pointer)
        end
      end

      def self.from_value(context, value, dtype: nil)
        case value
          when TensorHandle
            value
          when Data::Dataset
            value.variant_tensor
          when Tensor
            TensorHandle.new(context, value)
          when Variable
            value.value_handle
          else
            TensorHandle.new(context, Tensor.new(value, dtype: dtype))
        end
      end

      def initialize(context, value)
        @context = context
        case value
          when ::FFI::Pointer
            @pointer = value
          when Tensor
            Status.check do |status|
              @pointer = FFI.TFE_NewTensorHandle(value, status)
            end
            # We need to keep the tensor live so that it is not freed!
            @tensor = value
          else
            raise(Error::InvalidArgumentError, "Invalid value passed to tensor_handle: #{value}")
        end

        ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
      end

      def to_ptr
        @pointer
      end

      def tensor
        Status.check do |status|
          Tensor.from_pointer(FFI.TFE_TensorHandleResolve(self, status))
        end
      end

      def dtype
        FFI.TFE_TensorHandleDataType(self)
      end

      def element_count
        Status.check do |status|
          FFI.TFE_TensorHandleNumElements(self, status)
        end
      end

      def value
        self.tensor.value
      end

      private

      def num_dims
        Status.check do |status|
          FFI.TFE_TensorHandleNumDims(self, status)
        end
      end

      def dim(index)
        Status.check do |status|
          FFI.TFE_TensorHandleDim(self, index, status)
        end
      end
    end
  end
end