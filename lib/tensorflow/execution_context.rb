module Tensorflow
  class ExecutionContext
    class << self
      extend Forwardable
      def_delegators :context, :push, :pop, :current, :eager?, :graph?
    end

    def self.context
      Thread.current[:execution_context] ||= self.new
    end

    def initialize
      @stack = Array.new
    end

    def push(value)
      @stack.push(value)
    end

    def pop
      @stack.pop
    end

    def figure_from_inputs(inputs=[])
      inputs.flatten.each do |input|
        case input
          when Graph::Operation
            return input.graph
          when Eager::TensorHandle
            return input.context
        end
      end
      nil
    end

    def figure_from_context
      @stack.last
    end

    def figure_from_execution_mode
      if ::Tensorflow.execution_mode == Tensorflow::GRAPH_MODE
        Graph::Graph.default
      else
        Eager::Context.default
      end
    end

    def current(inputs=[])
      figure_from_context || figure_from_inputs(inputs) || figure_from_execution_mode
    end

    def eager?(inputs=[])
      context = self.current(inputs)
      context.is_a?(Eager::Context)
    end

    def graph?(inputs=[])
      context = self.current(inputs)
      context.is_a?(Graph::Graph)
    end
  end
end
