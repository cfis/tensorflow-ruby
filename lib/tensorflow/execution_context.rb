module Tensorflow
  class ExecutionContext
    def self.context
      Thread.current[:execution_context] ||= self.new
    end

    def push(value)
      @stack.push(value)
    end

    def pop
      @stack.pop
    end

    def initialize
      @stack = Array.new
    end

    def current
      if @stack.last
        @stack.last
      elsif ::Tensorflow.execution_mode == Tensorflow::GRAPH_MODE
        Graph::Graph.default
      else
        Eager::Context.default
      end
    end
  end
end
