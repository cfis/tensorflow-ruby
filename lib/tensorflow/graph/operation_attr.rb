module Tensorflow
  module Graph
    class OperationAttr
      attr_reader :metadata, :name, :operation

      def initialize(operation, name, metadata)
        @operation = operation
        @name = name
        @metadata = metadata
      end

      def list?
        self.metadata[:is_list] > 0
      end

      def value
        case self.metadata[:type]
          when :bool
            self.list? ? self.bool_list : self.bool
          when :int
            self.list? ? self.int_list : self.int
          when :float
            self.list? ? self.float_list : self.float
            self.float
          when :func
            self.list? ? self.func_list : self.func
          when :shape
            self.list? ? self.shape_list : self.shape
          when :string
            self.list? ? self.string_list : self.string
          when :tensor
            self.list? ? self.tensor_list : self.tensor
          when :type
            self.list? ? self.dtype_list : self.dtype
          else
            raise(Error::UnimplementedError, "Unsupported attribute. #{self.name} - #{self.metadata[:type]}")
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
        pointer = ::FFI::MemoryPointer.new(FFI::DataType.native_type)
        Status.check do |status|
          FFI.TF_OperationGetAttrType(self.operation, self.name, pointer, status)
        end
        value = pointer.read(FFI::DataType.native_type)
        FFI::DataType[value]
      end

      def dtype_list
        pointer = ::FFI::MemoryPointer.new(FFI::DataType.native_type, self.metadata[:list_size])
        Status.check do |status|
          FFI.TF_OperationGetAttrTypeList(self.operation, self.name, pointer, self.metadata[:list_size], status)
        end
        pointer.read_array_of_type(FFI::DataType.native_type, :read_uint32, self.metadata[:list_size]).map do |value|
          FFI::DataType[value]
        end
      end

      def float
        pointer = ::FFI::MemoryPointer.new(:float)
        Status.check do |status|
          FFI.TF_OperationGetAttrFloat(self.operation, self.name, pointer, status)
        end
        pointer.read_float
      end

      def func
        self.proto.func.name
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

      def shape_list
        total_size = self.metadata[:total_size]
        storage_ptr = ::FFI::MemoryPointer.new(:int64, total_size)
        dims_pointer = ::FFI::MemoryPointer.new(:pointer, self.metadata[:list_size])
        num_dims_pointer = ::FFI::MemoryPointer.new(:int, self.metadata[:list_size])
        Status.check do |status|
          FFI.TF_OperationGetAttrShapeList(self.operation, self.name,
                                           dims_pointer, num_dims_pointer,
                                           self.metadata[:list_size],
                                           storage_ptr, total_size, status)
        end

        num_dims = num_dims_pointer.read_array_of_int(self.metadata[:list_size])
        num_dims.map.with_index do |dims, i|
          shape_pointer = dims_pointer[i].read_pointer
          shape_pointer.read_array_of_int64(dims)
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

      def proto
        buffer_ptr = FFI.TF_NewBuffer
        Status.check do |status|
          FFI.TF_OperationGetAttrValueProto(self.operation, self.name, buffer_ptr, status)
        end
        buffer = FFI::Buffer.new(buffer_ptr)
        string = buffer[:data].read_string(buffer[:length])
        AttrValue.decode(string)
      ensure
        FFI.TF_DeleteBuffer(buffer)
      end

      def to_s
        "#{self.name}: #{self.value}"
      end
    end
  end
end
