module Tensorflow
  module Eager
    class TensorHandle
      include TensorMixin
      include Operators

      def self.finalize(pointer)
        proc do
          FFI.TFE_DeleteTensorHandle(pointer)
        end
      end

      def initialize(value)
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
            raise(TensorflowError, "Invalid value passed to tensor_handle: #{value}")
        end

        ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
      end

      def to_ptr
        @pointer
      end

      def tensor
        Status.check do |status|
          Tensor.new(FFI.TFE_TensorHandleResolve(self, status))
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