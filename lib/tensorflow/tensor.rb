module Tensorflow
  class Tensor
    include Operators
    include TensorMixin

    def self.finalize(pointer)
      proc do
        FFI.TF_DeleteTensor(pointer)
      end
    end

    def self.from_value(value)
      case value
        when Tensor
          value
        when Data::Dataset
          value.variant_tensor
        else
          Tensor.new(value)
      end
    end

    def self.from_pointer(pointer)
      result = self.allocate
      result.instance_variable_set(:@pointer, pointer)
      ObjectSpace.define_finalizer(result, self.finalize(pointer))
      result
    end

    def initialize(value, dtype: nil, shape: [])
      value = case value
                when Numo::NArray
                  value
                when Array
                  # We convert all arrays to narrays. This makes it a lot easier to support multidimensional arrays
                  result = Numo::NArray.cast(value)
                else
                  TensorData.value_with_shape(value, shape)
              end

      tensor_data = TensorData.new(value, dtype: dtype, shape: shape)
      dtype = tensor_data.dtype
      shape = tensor_data.shape

      if shape && shape.size > 0
        dims_ptr = ::FFI::MemoryPointer.new(:int64, shape.size)
        dims_ptr.write_array_of_int64(shape)
      else
        dims_ptr = nil
      end

      @pointer = FFI.TF_NewTensor(FFI::DataType[dtype],
                                  dims_ptr, shape ? shape.size : 0,
                                  tensor_data, tensor_data.byte_size,
                                  TensorData::Deallocator, nil)

      ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
    end

    def value
      self.data.value
    end

    def dtype
      FFI.TF_TensorType(self)
    end

    def to_s
      inspect
    end

    def to_ptr
      @pointer
    end

    def byte_size
      FFI.TF_TensorByteSize(self)
    end

    def inspect
      inspection = %w(numo shape dtype).map { |v| "#{v}: #{send(v).inspect}"}
      "#<#{self.class} #{inspection.join(", ")}>"
    end

    def data
      TensorData.from_pointer(FFI.TF_TensorData(self), self.byte_size, self.dtype, self.shape)
    end

    private

    def num_dims
      FFI.TF_NumDims(self)
    end

    def dim(index)
      FFI.TF_Dim(self, index)
    end

    def element_count
      FFI.TF_TensorElementCount(self)
    end

    def calculate_shape(value)
      return value.shape if value.respond_to?(:shape)

      shape = []
      d = value
      while d.is_a?(Array)
        shape << d.size
        d = d.first
      end
      shape
    end
  end
end