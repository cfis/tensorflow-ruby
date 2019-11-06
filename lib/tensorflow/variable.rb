module Tensorflow
  class Variable
    attr_reader :handle, :dtype

    def initialize(initial_value = nil, dtype: nil, shape: [], name: 'Variable')
      # We immediately convert to a tensor because otherwise dtypes get screwed up
      tensor = Tensor.from_value(initial_value)
      @dtype = tensor.dtype
      unique_name = ExecutionContext.context.current.unique_name(name)
      @handle = RawOps.var_handle_op(dtype: @dtype, shape: tensor.shape, shared_name: unique_name)
      self.value = initial_value
    end

    def value_handle
      RawOps.read_variable_op(self.handle, dtype: @dtype)
    end

    def value=(value)
      if value
        RawOps.assign_variable_op(self.handle, value, dtype: @dtype)
      end

      self
    end

    def shape
      raise(TensorflowError, "Only supported in eager execution mode") if Tensorflow.execution_mode == Tensorflow::GRAPH_MODE
      self.value_handle.shape
    end

    def tensor
      raise(TensorflowError, "Only supported in eager execution mode") if Tensorflow.execution_mode == Tensorflow::GRAPH_MODE
      self.value_handle.tensor
    end

    def value
      raise(TensorflowError, "Only supported in eager execution mode") if Tensorflow.execution_mode == Tensorflow::GRAPH_MODE
      self.value_handle.value
    end

    def rank
      self.shape.size
    end

    def reshape(shape)
      RawOps.reshape(self, shape)
    end

    def assign_add(value)
      RawOps.assign_add_variable_op(self.handle, value)
      self
    end

    def assign_sub(value)
      RawOps.assign_sub_variable_op(self.handle, value, dtype: self.dtype)
      self
    end

    def +(other)
      v = Variable.new(value, dtype: self.dtype)
      v.assign_add(other)
    end

    def -(other)
      v = Variable.new(value, dtype: self.dtype)
      v.assign_sub(other)
    end

    def to_s
      inspect
    end

    def inspect
      value = value_handle
      inspection = %w(shape dtype).map { |v| "#{v}: #{value.send(v).inspect}"}
      "#<#{self.class} #{inspection.join(", ")}>"
    end
  end
end
