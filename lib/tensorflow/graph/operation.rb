module Tensorflow
  module Graph
    class Operation
      include Operators
      attr_reader :graph

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

      def op_def
        self.graph.op_def(self.op_type)
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

      def to_output
        output = FFI::Output.new
        output[:oper] = self
        output[:index]  = 0
        output
      end

      def num_inputs
        FFI.TF_OperationNumInputs(self)
      end

      def inputs
        pointer = ::FFI::MemoryPointer.new(FFI::Output, self.num_inputs)
        FFI.TF_OperationAllInputs(self, pointer, self.num_inputs)
        self.num_inputs.times.map do |index|
          FFI::Output.new(pointer[index])
        end
      end

      def num_control_outputs
        FFI.TF_OperationNumControlOutputs(self)
      end

      def control_outputs
        pointer = ::FFI::MemoryPointer.new(:pointer, self.num_control_outputs)
        FFI.TF_OperationGetControlOutputs(self, pointer, self.num_control_outputs)
        self.num_control_outputs.times.map do |index|
          operation_ptr = pointer[index].read_pointer
          self.class.new(self.graph, operation_ptr)
        end
      end

      def num_outputs
        FFI.TF_OperationNumOutputs(self)
      end

      def outputs
        self.num_outputs.times.map do |i|
          output = FFI::Output.new
          output[:oper] = self
          output[:index] = i
          output
        end
      end

      def [](index)
        self.outputs[index]
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

      def num_control_inputs
        FFI.TF_OperationNumControlInputs(self)
      end

      def control_inputs
        pointer = ::FFI::MemoryPointer.new(:pointer, self.num_control_inputs)
        FFI.TF_OperationGetControlInputs(self, pointer, self.num_control_inputs)
        self.num_control_inputs.times.map do |index|
          operation_ptr = pointer[index].read_pointer
          self.class.new(self.graph, operation_ptr)
        end
      end

      def attributes
        self.op_def.attr.map do |attr_def|
          self.attr(attr_def.name)
        end
      end

      def attr(attr_name)
        metadata = Status.check do |status|
          FFI.TF_OperationGetAttrMetadata(self, attr_name, status)
        end

        OperationAttr.new(self, attr_name, metadata)
      end

      def output_consumers(index)
        # How many consumers does this output have?
        output = FFI::Output.new
        output[:oper] = self
        output[:index] = index
        count = FFI.TF_OperationOutputNumConsumers(output)

        # Get the consumers
        consumers_ptr = ::FFI::MemoryPointer.new(FFI::Input, count)
        FFI.TF_OperationOutputConsumers(output, consumers_ptr, count)

        count.times.map do |i|
          FFI::Input.new(consumers_ptr[i])
        end
      end

      def consumers
        self.num_outputs.times.reduce(Array.new) do |result, index|
          result.concat(self.output_consumers(index))
          result
        end
      end

      def to_s
        "#{self.op_type}, name: #{self.name}"
      end
    end
  end
end
