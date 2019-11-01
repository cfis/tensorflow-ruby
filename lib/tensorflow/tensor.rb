module Tensorflow
  class Tensor
    include Operators
    include TensorMixin

    def self.finalize(pointer)
      proc do
        FFI.TF_DeleteTensor(pointer)
      end
    end

    def initialize(value = nil, dtype: nil, shape: nil)
      value = Utils.reshape(value, dtype, shape)
      dtype ||= Utils.infer_dtype(value)
      case value
        when ::FFI::Pointer
          @pointer = value
          ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
          return
        when String
          data = TensorData.write_string(value)
        when Numo::RObject
          # Let's assume RObjects are always arrays of strings
          shape ||= value.shape
          data = TensorData.write_array_of_string(value.to_a)
        when Numo::NArray
          # TODO - validate shape or reshape
          shape ||= value.shape

          # Hack for boolean types
          if dtype == :bool
            value = value.cast_to(Numo::Int8)
          elsif Utils::NUMO_TYPE_TO_DTYPE_MAP[value.class] != dtype
            # Cast the narray if necessary (say the user passed in float but we have a double array)
            value = value.cast_to(Utils::DTYPE_TO_NUMO_TYPE_MAP[dtype])
          end

          data = TensorData.new(:uchar, value.byte_size)
          data.write_bytes(value.to_binary)
        when Integer
          data = TensorData.new(dtype)
          data.send("write_#{dtype}", value)
        when Complex
          if dtype == :complex64
            data = TensorData.write_complex64(value)
          else
            data = TensorData.write_complex128(value)
          end
        when Numeric
          data = TensorData.new(dtype)
          data.send("write_#{dtype}", value)
        when TrueClass, FalseClass
          data = TensorData.new(:uchar)
          data.write_uchar(value ? 1 : 0)
        else
          raise(::TensorflowError, "Unsupported data type")
      end

      if shape && shape.size > 0
        dims_ptr = ::FFI::MemoryPointer.new(:int64, shape.size)
        dims_ptr.write_array_of_int64(shape)
      else
        dims_ptr = nil
      end

      @pointer = FFI.TF_NewTensor(FFI::DataType[dtype],
                                  dims_ptr, shape ? shape.size : 0,
                                  data, data.size,
                                  TensorData::Deallocator, nil)
      ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
    end

    def value
      # This would be a nice approach but doesn't seem to always work. See https://github.com/ruby-numo/numo-narray/issues/142
      # bytes = self.data.read_bytes(self.byte_size)
      # numo_klass = Utils::DTYPE_TO_NUMO_TYPE_MAP[self.dtype]
      # narray = if self.shape.empty?
      #           numo_klass.from_binary(bytes)
      #          else
      #            numo_klass.from_binary(bytes, self.shape)
      #          end
      #
      # if element_count == 1
      #   narray.to_a.first
      # else
      #   narray
      # end

      value = case dtype
                when :float, :double, :int32, :uint8, :int16, :int8, :int64, :uint16, :uint32, :uint64
                  self.data.send("read_array_of_#{dtype}", element_count)
                when :bfloat16
                  byte_str = self.data.read_bytes(element_count * 2)
                  element_count.times.map { |i| "#{byte_str[(2 * i)..(2 * i + 1)]}\x00\x00".unpack1("g") }
                when :complex64
                  self.data.read_array_of_complex64(element_count)
                when :complex128
                  self.data.read_array_of_complex128(element_count)
                when :string
                  self.data.read_array_of_string(element_count)
                when :bool
                  self.data.read_array_of_int8(element_count)
                when :resource, :variant
                  return self.data
                else
                  raise "Unknown type: #{dtype}"
                end

      if element_count == 1
        value.first
      elsif self.shape.length == 1
        value
      else
        numo_klass = Utils::DTYPE_TO_NUMO_TYPE_MAP[self.dtype]
        numo_klass[value].reshape(*self.shape)
      end
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
      TensorData.from_pointer(FFI.TF_TensorData(self), byte_size, element_count)
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