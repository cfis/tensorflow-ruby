module Tensorflow
  class Variable
    include Operators

    attr_reader :handle, :dtype, :name

    def initialize(initial_value = nil, dtype: nil, shape: nil, shared_name: nil, name: 'Variable', trainable: true)
      initial_value = case initial_value
                      when NilClass
                        @dtype = dtype
                        shape = []
                        initial_value
                      when Graph::Operation
                        @dtype = dtype || initial_value.dtype
                        shape = shape || initial_value.output_shapes.first
                        initial_value
                      when Tensor
                        @dtype = initial_value.dtype
                        shape = shape || initial_value.shape
                        initial_value
                      else
                        tensor = Tensor.from_value(initial_value, dtype: dtype)
                        @dtype = tensor.dtype
                        shape = tensor.shape
                        tensor
                      end

      name = name&.to_s
      shared_name = shared_name&.to_s
      unique_name = ExecutionContext.current.unique_name(name || shared_name)
      shared_name ||= unique_name
      @name = unique_name

      collections = [Graph::GraphKeys::GLOBAL_VARIABLES]
      if trainable
        collections << Graph::GraphKeys::TRAINABLE_VARIABLES
      end

      ExecutionContext.current.add_to_collections(collections, self)

      @handle = RawOps.var_handle_op(dtype: @dtype, shape: shape, shared_name: shared_name, name: unique_name)
      self.value = initial_value if initial_value
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

    # These methods match the operation api to enable gradients and sessions
    def consumers
      self.handle.consumers
    end

    # This enables executing variables to get the values in a session
    def outputs
      [Graph::OperationOutput.from_index(self.value_handle, 0)]
    end

    def to_ptr
      self.handle.to_ptr
    end

    def shape
      self.value_handle.shape
    end

    def tensor
      raise(Error::UnavailableError, "Only supported in eager execution mode") if Tensorflow.execution_mode == Tensorflow::GRAPH_MODE
      self.value_handle.tensor
    end

    def rank
      self.shape.size
    end

    def reshape(shape)
      RawOps.reshape(self, shape)
    end

    def assign_add(value, dtype: nil)
      @value_handle = nil
      tensor = Tensor.from_value(value, dtype: dtype)
      tensor = Tensorflow.cast(tensor, self.dtype)
      RawOps.assign_add_variable_op(self.handle, value, dtype: tensor.dtype)
    end

    def assign_sub(value)
      @value_handle = nil
      tensor = Tensor.from_value(value, dtype: dtype)
      tensor = Tensorflow.cast(tensor, self.dtype)
      RawOps.assign_sub_variable_op(self.handle, value, dtype: tensor.dtype)
    end

    def to_s
      inspect
    end

    def inspect
      inspection = []
      inspection << ["name: #{self.handle.name}"] if self.handle.respond_to?(:name)
      inspection << ["shape: #{self.value_handle.shape}"]
      inspection << ["dtype: #{self.value_handle.dtype}"]
      "#<#{self.class} #{inspection.join(", ")}>"
    end
  end
end
