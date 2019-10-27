module Tensorflow
  module Graph
    class Operation
      def self.op_defs
        buffer = FFI.TF_GetAllOpList
        string = buffer[:data].read_string(buffer[:length])
        ops = OpList.decode(string)
        ops.op.each_with_object(Hash.new) do |op_def, hash|
          hash[op_def.name] = op_def
        end
      ensure
        FFI.TF_DeleteBuffer(buffer)
      end

      def self.op_def(op_name)
        self.op_defs[op_name]
      end

      def initialize(pointer)
        @pointer = pointer
      end

      def to_ptr
        @pointer
      end

      def name
        FFI.TF_OperationName(self)
      end

      def op_type
        FFI.TF_OperationOpType(self)
      end

      def device
        FFI.TF_OperationDevice(self)
      end

      def consumers
        output = FFI::Output.new
        output[:oper] = self
        output[:index] = 0

        count = FFI.TF_OperationOutputNumConsumers(output)
        consumers = ::FFI::MemoryPointer.new(FFI::Output, count)
        FFI.TF_OperationOutputConsumers(output, consumers, count)

        result = Array.new
        count.times do |i|
          pointer = consumers[i]
          input = FFI::Input.new(pointer)
          result << Operation.new(input[:oper])
        end
        result
      end

      def num_outputs
        FFI.TF_OperationNumOutputs(self)
      end

      def output_type
         output = FFI::Output.new
         output[:oper] = self.to_ptr
         output[:index] = 0
         FFI.TF_OperationOutputType(output)
      end

      def output_list_length(arg_name)
        Status.check do |status|
          FFI.TF_OperationOutputListLength(self, arg_name, status)
        end
      end

      def attr(attr_name)
        metadata = Status.check do |status|
          FFI.TF_OperationGetAttrMetadata(self, attr_name, status)
        end

        OperationAttr.new(self, attr_name, metadata)
      end

      def eql?(other)
        self.name.eql?(other.name)
      end

      def ==(other)
        self.name == other.name
      end
    end

    class OperationAttr
      attr_reader :metadata, :name, :operation
      def initialize(operation, name, metadata)
        @operation = operation
        @name = name
        @metadata = metadata
      end

      def dtype
        pointer = ::FFI::MemoryPointer.new(:uint8)
        Status.check do |status|
          FFI.TF_OperationGetAttrType(self.operation, self.name, pointer, status)
        end
        FFI::DataType[pointer.read_uint8]
      end

      def shape
        size = self.metadata[:total_size]
        pointer = ::FFI::MemoryPointer.new(:int64, size)
        Status.check do |status|
          FFI.TF_OperationGetAttrShape(self.operation, self.name, pointer, size, status)
        end
        pointer.read_array_of_int64(size)
      end

      def tensor
        size = self.metadata[:total_size]
        pointer = ::FFI::MemoryPointer.new(:pointer)
        Status.check do |status|
          FFI.TF_OperationGetAttrTensor(self.operation, self.name, pointer, status)
        end
        tensor_pointer = pointer.read_pointer
        Tensor.new(:pointer => tensor_pointer)
      end
    end
  end
end
