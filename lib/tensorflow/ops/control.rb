module Tensorflow
  module Control
    def self.group(inputs, attrs={})
      graph = ExecutionContext.current(inputs)
      return if graph.is_a?(Eager::Context)

      graph.control_dependencies(inputs) do
        RawOps.no_op
      end
    end
  end
end

