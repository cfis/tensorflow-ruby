module Tensorflow
  module OpsExecutor
    def self.figure_graph_or_context(inputs)
      if inputs.empty?
        Eager::Context.default
      elsif inputs.flatten.first.is_a?(Graph::Operation)
        inputs.flatten.first.graph
      else
        Eager::Context.default
      end
    end

    def self.execute(op_name, inputs=[], attrs={})
      executor = self.figure_graph_or_context(inputs)
      operation = executor.create_operation(op_name, inputs, attrs)
      if executor.is_a?(Graph::Graph)
        operation
      else
        executor.execute(operation)
      end
    end
  end
end