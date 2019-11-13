module Tensorflow
  module Graph
    class OperationAttr
      attr_reader :metadata, :name, :operation

      def initialize(operation, name, metadata)
        @operation = operation
        @name = name
        @metadata = metadata
      end

      def value
        case self.metadata[:type]
          when :bool
            self.bool
          when :int
            self.int
          when :float
            self.float
          when :func
            self.func
          when :shape
            self.shape
          when :string
            self.string
          when :tensor
            self.tensor
          when :type
            self.dtype
          else
            raise(TensorflowError, "Unsupported attribute. #{self.name} - #{self.metadata[:type]}")
        end
      end

      def bool
        pointer = ::FFI::MemoryPointer.new(:uchar)
        Status.check do |status|
          FFI.TF_OperationGetAttrBool(self.operation, self.name, pointer, status)
        end
        Boolean(pointer.read_uchar)
      end

      def dtype
        pointer = ::FFI::MemoryPointer.new(:uint8)
        Status.check do |status|
          FFI.TF_OperationGetAttrType(self.operation, self.name, pointer, status)
        end
        FFI::DataType[pointer.read_uint8]
      end

      def float
        pointer = ::FFI::MemoryPointer.new(:float)
        Status.check do |status|
          FFI.TF_OperationGetAttrFloat(self.operation, self.name, pointer, status)
        end
        pointer.read_float
      end

      def func
        pointer = ::FFI::MemoryPointer.new(:float)
        Status.check do |status|
          FFI.TF_OperationGetAttrFloat(self.operation, self.name, pointer, status)
        end
        pointer.read_float
      end

      def int
        pointer = ::FFI::MemoryPointer.new(:int64)
        Status.check do |status|
          FFI.TF_OperationGetAttrInt(self.operation, self.name, pointer, status)
        end
        pointer.read_int
      end

      def shape
        size = self.metadata[:total_size]
        if size == -1
          []
        else
          pointer = ::FFI::MemoryPointer.new(:int64, size)
          Status.check do |status|
            FFI.TF_OperationGetAttrShape(self.operation, self.name, pointer, size, status)
          end
          pointer.read_array_of_int64(size)
        end
      end

      def string
        size = self.metadata[:total_size]
        pointer = ::FFI::MemoryPointer.new(:string, size)
        Status.check do |status|
          FFI.TF_OperationGetAttrString(self.operation, self.name, pointer, size, status)
        end
        pointer.read_string
      end

      def tensor
        pointer = ::FFI::MemoryPointer.new(:pointer)
        Status.check do |status|
          FFI.TF_OperationGetAttrTensor(self.operation, self.name, pointer, status)
        end
        Tensor.from_pointer(pointer.read_pointer)
      end
    end
  end
end
