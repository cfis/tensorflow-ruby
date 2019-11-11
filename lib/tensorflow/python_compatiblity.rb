module Tensorflow
  module PythonCompatability
    def disable_eager_execution
      self.execution_mode = Tensorflow::GRAPH_MODE
    end

    def enable_eager_execution
      self.execution_mode = Tensorflow::EAGER_MODE
    end

    def global_variables
      if ExecutionContext.eager?
        []
      else
        ExecutionContext.current.get_collection_ref(Graph::GraphKeys::GLOBAL_VARIABLES)
      end
    end

    def global_variables_initializer
      if ExecutionContext.eager?
        RawOps.no_op
      else
        global_variables = ExecutionContext.current.get_collection_ref(Graph::GraphKeys::GLOBAL_VARIABLES)
        global_variables = Array(global_variables)
        if global_variables.length > 0
          self.variables_initializer(global_variables)
        end
      end
    end

    def variables_initializer(variables, name: 'init')
      if ExecutionContext.eager?
        RawOps.no_op
      else
        Control.group(variables.map(&:initializer))
      end
    end
  end
end

Tf = Tensorflow
