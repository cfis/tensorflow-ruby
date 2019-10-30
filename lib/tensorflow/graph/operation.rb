module Tensorflow
  module Graph
    class Operation
      attr_reader :graph

      def self.create(graph, op_type, name, *inputs, **attrs)
        op_desc = OperationDescription.new(graph, op_type, name, inputs, attrs)
        op_desc.save
      end

      def initialize(graph, pointer)
        @graph = graph
        @pointer = pointer
      end

      def to_ptr
        @pointer
      end

      def eql?(other)
        self.name.eql?(other.name)
      end

      def ==(other)
        self.name == other.name
      end

      def hash
        self.name.hash
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

      def node_def
        buffer = FFI::Buffer.new
        Status.check do |status|
          FFI.TF_OperationToNodeDef(self, buffer, status)
        end
        string = buffer[:data].read_string(buffer[:length])
        ops = NodeDef.decode(string)
      end

      def num_inputs
        FFI.TF_OperationNumInputs(self)
      end

      def inputs
        result = Array.new

        pointer = ::FFI::MemoryPointer.new(FFI::Output, self.num_inputs)
        FFI.TF_OperationAllInputs(self, pointer, self.num_inputs)
        self.num_inputs.times do |index|
          current_pointer = pointer + (FFI::Output.size * index)
          output = FFI::Output.new(current_pointer)
          result << self.class.new(self.graph, output[:oper])
        end
        result
      end

      def num_outputs
        FFI.TF_OperationNumOutputs(self)
      end

      def output_types
        result = Array.new(self.num_outputs)
        self.num_outputs.times do |index|
          output = FFI::Output.new
          output[:oper] = self.to_ptr
          output[:index] = index
          result[index] = FFI.TF_OperationOutputType(output)
        end
        result
      end

      def output_list_length(arg_name)
        Status.check do |status|
          FFI.TF_OperationOutputListLength(self, arg_name, status)
        end
      end

      def attributes
        op_def = self.graph.op_def(self.op_type)
        op_def.attr.map do |attr_def|
          self.attr(attr_def.name)
        end
      end

      def attr(attr_name)
        metadata = Status.check do |status|
          FFI.TF_OperationGetAttrMetadata(self, attr_name, status)
        end

        OperationAttr.new(self, attr_name, metadata)
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
          result << Operation.new(self.graph, input[:oper])
        end
        result
      end

      def to_s
        "#{self.op_type}, name: #{self.name}"
      end
    end

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
        pointer.read_uchar == 1 ? true : false
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
        tensor_pointer = pointer.read_pointer
        Tensor.new(:pointer => tensor_pointer)
      end
    end
  end
end
