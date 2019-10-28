module Tensorflow
  module Graph
    class Gradients
      attr_reader :graph

      def initialize(graph)
        @graph = graph
      end

      def find_operations(output, input)
        result = [input]
        input.consumers.each do |consumer|
          if consumer == output || consumer.consumers.include?(output)
            result << consumer
          end
        end
        result
      end

      def derivatives(output, inputs, name: "gradients", stop_gradients: nil)
        inputs.map do |input|
          #stops = stop_gradients ? stop_gradients.map(&:name).join("_") : ""
          #gradient_program_name = "grad_#{tensor_ys.name}_#{x.name}_#{stops}".to_sym
          #tensor_graph = tensor_ys.graph

          # Next create new operations in the graph to calculate the derivatives for each operation
          self.derivative_operations(output, input)
        end
      end

      def derivative(output, input)
        # First find all operations between the output and input operation
        operations = self.find_operations(output, input)

        shape_op = Tensorflow.shape(output, :int64)
        constant = graph.constant(1, 'one')

        fill_op = Tensorflow.fill(shape_op, constant)
      end
    end
  end
end