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
        buffer_ptr = FFI.TF_NewBuffer
        Status.check do |status|
          FFI.TF_OperationToNodeDef(self, buffer_ptr, status)
        end
        buffer = FFI::Buffer.new(buffer_ptr)
        string = buffer[:data].read_string(buffer[:length])
        NodeDef.decode(string)
      ensure
        FFI.TF_DeleteBuffer(buffer)
      end

      def num_inputs
        FFI.TF_OperationNumInputs(self)
      end

      def inputs
        pointer = ::FFI::MemoryPointer.new(FFI::Output, self.num_inputs)
        FFI.TF_OperationAllInputs(self, pointer, self.num_inputs)
        self.num_inputs.times.map do |index|
          OperationOutput.from_graph(self.graph, pointer[index])
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
          OperationOutput.from_index(self, i)
        end
      end

      def [](index)
        self.outputs[index]
      end

      def output_types
        self.outputs.map do |output|
          FFI.TF_OperationOutputType(output)
        end
      end

      def output_shapes
        self.graph.output_shapes(self)
      end

      def shape
        self.output_shapes.first
      end

      def dtype
        self.output_types.first
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

      def output_consumers(output)
        # How many consumers does this output have?
        count = FFI.TF_OperationOutputNumConsumers(output)

        # Get the consumers
        consumers_ptr = ::FFI::MemoryPointer.new(FFI::Input, count)
        FFI.TF_OperationOutputConsumers(output, consumers_ptr, count)

        count.times.map do |i|
          OperationOutput.from_graph(self.graph, consumers_ptr[i])
        end
      end

      def consumers
        self.outputs.reduce(Array.new) do |result, output|
          result.concat(self.output_consumers(output))
          result
        end
      end

      def to_s
        result = [self.op_type]
        result << "name=#{self.name}"
        outputs.length.times do |index|
          result << "#{index}:(shape=#{self.output_shapes[index]}, dtype=#{self.output_types[index]})"
        end
        result.join(', ')
      end
    end
  end
end
