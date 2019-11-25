module Tensorflow
  module Graph
    class OperationOutput
      attr_reader :operation, :output

      def self.from_pointer(operation, pointer)
        output = FFI::Output.new(pointer)
        self.new(operation, output)
      end

      def self.from_index(operation, index)
        output = FFI::Output.new
        output[:index] = index
        output[:oper] = operation
        self.new(operation, output)
      end

      def self.from_graph(graph, pointer)
        output = FFI::Output.new(pointer)
        operation = Operation.new(graph, output[:oper])
        self.new(operation, output)
      end

      def initialize(operation, output)
        @operation = operation
        @output = output
      end

      def to_ptr
        @output.to_ptr
      end

      def index
        self.output[:index]
      end

      def to_s
        if self.output
          result = [self.operation.op_type]
          result << "name=#{self.operation.name}"
          result << "#{self.index}:(shape=#{self.operation.output_shapes[self.index]}, dtype=#{self.operation.output_types[self.index]})"
          result.join(', ')
        else
          super
        end
      end
    end
  end
end
