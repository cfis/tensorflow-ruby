module Tensorflow
  module PythonCompatability
    def disable_eager_execution
      self.execution_mode = Tensorflow::GRAPH_MODE
    end

    def enable_eager_execution
      self.execution_mode = Tensorflow::EAGER_MODE
    end
  end
end

Tf = Tensorflow
