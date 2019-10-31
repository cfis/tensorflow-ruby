module Tensorflow
  module Utils
    DTYPE_TO_NUMO_TYPE_MAP = {bool:   Numo::Bit,
                              double: Numo::DFloat,
                              float:  Numo::SFloat,
                              int8:   Numo::Int8,
                              int16:  Numo::Int16,
                              int32:  Numo::Int32,
                              int64:  Numo::Int64,
                              uint8:  Numo::UInt8,
                              uint16: Numo::UInt16,
                              uint32: Numo::UInt32,
                              uint64: Numo::UInt64,
                              string: Numo::RObject}

    NUMO_TYPE_TO_DTYPE_MAP = DTYPE_TO_NUMO_TYPE_MAP.each_with_object(Hash.new) do |pair, hash|
                               hash[pair.last] = pair.first
                             end

    def self.infer_dtype(value)
      case value
        when Numo::NArray
          NUMO_TYPE_TO_DTYPE_MAP[value.class]
        when Integer
          (value >= -2147483648 && value <= 2147483647) ? :int32 : :int64
        when Complex
          :complex128
        when Numeric
          :float
        when String
          :string
        when TrueClass, FalseClass
          :bool
        when ::FFI::Pointer
          :pointer
        else
          raise(::TensorflowError, "Unsupported type: #{value.class}")
      end
    end

    def self.infer_numo_type(value)
      dtype = self.infer_dtype(value)
      DTYPE_TO_NUMO_TYPE_MAP[dtype]
    end

    def self.to_tensor_array(values)
      case values
        when Numo::NArray
          [Tensor.new(values)]
        when Tensor
          [values]
        else
          values.to_a.map do |v|
            if v.is_a?(Tensor)
              v
            else
              Tensor.new(v)
            end
          end
      end
    end
  end
end
