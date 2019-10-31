require 'forwardable'

module Tensorflow
  class Variable
    extend Forwardable
    def_delegators :value_handle, :dtype, :shape, :tensor, :value
    attr_reader :name, :handle

    def initialize(initial_value = nil, dtype: nil, shape: nil, name: nil)
      # We convert all arrays to narrays. This makes it a lot easier to support multidimensional arrays
      initial_value = Numo::NArray.cast(initial_value) if initial_value.is_a?(Array)

      @dtype = dtype || Utils.infer_dtype(initial_value)
      @shape = shape
      @name = name
      @handle = RawOps.var_handle_op(dtype: type_enum, shape: [], shared_name: Eager::Context.default.shared_name)
      self.value = initial_value
    end

    def value_handle
      RawOps.read_variable_op(self.handle, dtype: type_enum)
    end

    def value=(value)
      if value
        value = Eager.convert_to_tensor_handle(value, dtype: @dtype)
        RawOps.assign_variable_op(self.handle, value)
      end

      self
    end

    def assign_add(value)
      value = Eager.convert_to_tensor_handle(value, dtype: @dtype)
      RawOps.assign_add_variable_op(self.handle, value)
      self
    end

    def assign_sub(value)
      value = Eager.convert_to_tensor_handle(value, dtype: @dtype)
      RawOps.assign_sub_variable_op(self.handle, value)
      self
    end

    def +(other)
      v = Variable.new(value, dtype: @dtype)
      v.assign_add(other)
    end

    def -(other)
      v = Variable.new(value, dtype: @dtype)
      v.assign_sub(other)
    end

    def to_s
      inspect
    end

    def rank
      self.shape.size
    end

    def reshape(shape)
      RawOps.reshape(self, shape)
    end

    def inspect
      value = value_handle
      inspection = %w(shape dtype).map { |v| "#{v}: #{value.send(v).inspect}"}
      inspection.unshift("name: #{name}") if name
      "#<#{self.class} #{inspection.join(", ")}>"
    end

    private

    def type_enum
      FFI::DataType[@dtype.to_sym] if @dtype
    end
  end
end
