require 'set'

module Tensorflow
  module Graph
    class Gradients
      attr_reader :graph

      def initialize(graph)
        @graph = graph
      end

      def path(output, input)
        forwards = self.graph.forward(input)
        backwards = self.graph.backward(output)
        forwards.intersection(backwards)
      end

      def derivatives(output, inputs, name: "gradients", stop_operations: Set.new)
        self.graph.name_scope(name) do
          inputs.map.with_index do |input, i|
            operations_path = self.path(output, input)

            shape_op = Tensorflow.shape(output, :int32)
            const_name = "grad_ys_#{i}"
            const_type = output.output_types.first
            constant = graph.constant(1, name: const_name, dtype: const_type)
            fill_op = Tensorflow.fill(shape_op, constant)

            self.derivative(fill_op, output, stop_operations, operations_path)
          end
        end.compact.flatten
      end

      def derivative(gradient, operation, stop_operations, operations_path)
        input_operations = operation.inputs.select do |input_operation|
          operations_path.include?(input_operation) && !stop_operations.include?(input_operation)
        end

        return gradient if input_operations.empty?

        # This is the output operation
        y = FFI::Output.new
        y[:oper] = operation

        # Setup the input operations
        x = ::FFI::MemoryPointer.new(FFI::Output, input_operations.length)
        input_operations.each_with_index do |input, index|
          output = FFI::Output.new(x[index])
          output[:oper] = input
          output[:index] = index
        end

        # This is the gradient
        dx = FFI::Output.new
        dx[:oper] = gradient

        # This is the change in y
        dy = ::FFI::MemoryPointer.new(FFI::Output, input_operations.length, true)

        Status.check do |status|
          FFI.TF_AddGradients(self.graph,
                              y, 1,
                              x, input_operations.length,
                              dx, status, dy)
        end

        input_operations.map.with_index do |input_operation, i|
          dy_output = FFI::Output.new(dy[i])
          unless dy_output[:oper].null?
            dy_operation = Operation.new(self.graph, dy_output[:oper])
            self.derivative(dy_operation, input_operation, stop_operations, operations_path)
          end
        end
      end
    end
  end
end