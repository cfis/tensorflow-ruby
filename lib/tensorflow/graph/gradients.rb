require 'set'

module Tensorflow
  module Graph
    class Gradients
      attr_reader :graph

      def self.gradients
        @gradients ||= begin
          default = self.instance_method(:add_api_gradients)
          Hash.new(default)
        end
      end

      def self.register(op_type, &block)
        self.gradients[op_type] = block
      end

      def initialize(graph)
        @graph = graph
      end

      def path(output, input)
        forwards = self.graph.forward(input)
        backwards = self.graph.backward(output)
        forwards.intersection(backwards)
      end

      def default_gradient(operation)
        operation.outputs.map.with_index do |output, i|
          shape_op = Tensorflow.shape(output, :int32)
          constant = Tensorflow.constant(1, name: "grad_ys_#{i}", dtype: operation.output_types[i])
          Tensorflow.fill(shape_op, constant)
        end
      end

      def gradients(output, inputs, grad_ys: nil, name: "gradients", stop_operations: Set.new)
        grad_ys ||= default_gradient(output).first

        self.graph.name_scope(name) do
          inputs.map.with_index do |input, i|
            operations_path = self.path(output, input)
            next if operations_path.empty?

            self.derivative(grad_ys, output, stop_operations, operations_path)
          end.flatten.compact
        end
      end

      def derivative(gradient, operation, stop_operations, operations_path)
        # This method follows the C api naming conventions for parameters. Visually it looks
        # like this:
        #
        #   x  ------>  y  (forward)
        #   dy <-----   dx (backward)

        return gradient if !operations_path.include?(operation) || stop_operations.include?(operation)

        inputs = operation.inputs.select do |input|
          operations_path.include?(input.operation) && !stop_operations.include?(input.operation)
        end

        return gradient if inputs.empty?

        outputs = operation.outputs.select do |output|
          consumers = operation.output_consumers(output)
          # The last operation we are evaluating will not be hooked up to any consumers, so
          # we want to analyze all its outputs. For operations earlier in the graph, skip any
          # unused outputs since they are not connected to anything
          operation == operations_path.first || consumers.count > 0
        end

        gradient_func = self.class.gradients[operation.op_type]

        dy = if gradient_func.is_a?(UnboundMethod)
               gradient_func.bind(self).call(gradient, outputs, inputs)
             else
               gradient_func.call(gradient, outputs, inputs)
             end

        # We are done with this operation, so backpropagate to the input operations
        inputs.map.with_index do |input, i|
          dy_output = dy[i]
          unless dy_output.output[:oper].null?
            self.derivative(dy_output.operation, input.operation, stop_operations, operations_path)
          end
        end
      end

      def add_api_gradients(gradient, outputs, inputs)
        # These are the outputs from the operation
        y = FFI::Output.array_to_ptr(outputs.map(&:output))

        # These are the inputs to the output operation
        x = FFI::Output.array_to_ptr(inputs.map(&:output))

        # This is the gradient we are backpropagating
        dx = if gradient
               FFI::Output.array_to_ptr(gradient.outputs.map(&:output))
             end

        # This is the gradient we want to calculate
        dy = ::FFI::MemoryPointer.new(FFI::Output, inputs.length, true)

        Status.check do |status|
          FFI.TF_AddGradients(self.graph,
                              y, outputs.length,
                              x, inputs.length,
                              dx, status, dy)
        end

        inputs.length.times.map do |i|
          OperationOutput.from_graph(graph, dy[i])
        end
      end
    end
  end
end