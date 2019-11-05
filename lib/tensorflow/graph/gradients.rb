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

      def gradients(output, inputs, name: "gradients", stop_operations: Set.new)
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
        # This method follows the C api naming conventions for parameters. Visually it looks
        # like this:
        #
        #   x  ------>  y  (forward)
        #   dy <-----   dx (backward)

        inputs = operation.inputs.select do |input|
          input_operation = input.operation(self.graph)
          operations_path.include?(input_operation) && !stop_operations.include?(input_operation)
        end

        outputs = operation.outputs.select do |output|
          output_operation = output.operation(self.graph)
          operations_path.include?(output_operation) && !stop_operations.include?(output_operation)
        end

        return gradient if inputs.empty?

        # These are the outputs from the operation
        y = FFI::Output.array_to_ptr(outputs)

        # These are the inputs to the output operation
        x = FFI::Output.array_to_ptr(inputs)

        # This is the gradient we are backpropagating
        dx = if gradient
               FFI::Output.array_to_ptr(gradient.outputs)
             end

        # This is the gradient we want to calculate
        dy = ::FFI::MemoryPointer.new(FFI::Output, inputs.length, true)

        Status.check do |status|
          FFI.TF_AddGradients(self.graph,
                              y, outputs.length,
                              x, inputs.length,
                              dx, status, dy)
        end

        # We are done with this operation, so backpropagate to the input operations
        inputs.map.with_index do |input, i|
          dy_output = FFI::Output.new(dy[i])
          unless dy_output[:oper].null?
            input_operation = Operation.new(self.graph, input[:oper])
            dy_operation = Operation.new(self.graph, dy_output[:oper])
            self.derivative(dy_operation, input_operation, stop_operations, operations_path)
          end
        end
      end
    end
  end
end