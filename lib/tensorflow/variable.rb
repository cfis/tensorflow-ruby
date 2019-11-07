module Tensorflow
  class Variable
    include Operators

    attr_reader :handle, :dtype

    def initialize(initial_value = nil, dtype: nil, shape: [], shared_name: nil, name: 'Variable', trainable: false)
      if initial_value
        # We immediately convert to a tensor because otherwise dtypes get screwed up
        tensor = Tensor.from_value(initial_value, dtype: dtype)
        @dtype = tensor.dtype
        shape = tensor.shape
      else
        @dtype = dtype
      end

      unique_name = ExecutionContext.current.unique_name(name || shared_name)
      shared_name ||= unique_name

      collections = [Graph::GraphKeys::GLOBAL_VARIABLES]
      if trainable
        collections << Graph::GraphKeys::TRAINABLE_VARIABLES
      end

      if ExecutionContext.current.is_a?(Graph::Graph)
        ExecutionContext.current.add_to_collections(collections, self)
      end

      @handle = RawOps.var_handle_op(dtype: @dtype, shape: shape, shared_name: shared_name, name: unique_name)
      self.value = tensor if tensor
    end

    def value_handle
      @value_handle ||= RawOps.read_variable_op(self.handle, dtype: @dtype)
    end

    def value
      case value_handle
        when Eager::TensorHandle
          value_handle.value
        when Graph::Operation
          value_handle
      end
    end

    def value=(value)
      @initializer = RawOps.assign_variable_op(self.handle, value, dtype: @dtype)
    end

    def initializer
      @initializer
    end

    def initialized?
      RawOps.var_is_initialized_op(self.handle)
    end

    # Pretend to be an operation to make implementating Session#run cleaner
    def outputs
      []
    end

    def to_ptr
      self.handle.to_ptr
    end

    def shape
      case self.handle
        when Eager::TensorHandle
          self.value_handle.shape
        else
          self.handle.graph.tensor_get_shape(self.handle)
      end
    end

    def tensor
      raise(TensorflowError, "Only supported in eager execution mode") if Tensorflow.execution_mode == Tensorflow::GRAPH_MODE
      self.value_handle.tensor
    end

    def rank
      self.shape.size
    end

    def reshape(shape)
      RawOps.reshape(self, shape)
    end

    def assign_add(value)
      RawOps.assign_add_variable_op(self.handle, value)
    end

    def assign_sub(value)
      RawOps.assign_sub_variable_op(self.handle, value, dtype: self.dtype)
    end

    # def to_s
    #   inspect
    # end
    #
    # def inspect
    #   value = value_handle
    #   inspection = %w(shape dtype).map { |v| "#{v}: #{value.send(v).inspect}"}
    #   "#<#{self.class} #{inspection.join(", ")}>"
    # end
  end
end
