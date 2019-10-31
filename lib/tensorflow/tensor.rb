module Tensorflow
  class Tensor
    include TensorMixin

    def self.finalize(pointer)
      proc do
        FFI.TF_DeleteTensor(pointer)
      end
    end

    def initialize(value = nil, dtype: nil, shape: nil)
      # We convert all arrays to narrays. This makes it a lot easier to support multidimensional arrays
      value = Numo::NArray.cast(value) if value.is_a?(Array)
      dtype ||= Utils.infer_dtype(value)
      case value
        when ::FFI::Pointer
          @pointer = value
          ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
          return
        when Numo::Bit
          # Let's assume bit arrays are always arrays of true/false
          shape ||= value.shape
          value = value.cast_to(Numo::Int8)
          data_ptr = TensorDataPointer.new(:uchar, value.byte_size)
          data_ptr.write_bytes(value.to_string)
        when Numo::RObject
          # Let's assume RObjects are always arrays of strings
          data_ptr = write_tensor_string(value.to_a)
        when Numo::NArray
          # TODO - validate shape or reshape
          shape ||= value.shape
          data_ptr = TensorDataPointer.new(:uchar, value.byte_size)
          data_ptr.write_bytes(value.to_string)
        when Integer
          data_ptr = TensorDataPointer.new(dtype)
          data_ptr.send("write_#{dtype}", value)
        when Complex
          if dtype == :complex64
            data_ptr = TensorDataPointer.new(:float, 2)
            data_ptr.write_array_of_float([value.real, value.imaginary])
          else
            data_ptr = TensorDataPointer.new(:double, 2)
            data_ptr.write_array_of_float([value.real, value.imaginary])
          end
        when Numeric
          data_ptr = TensorDataPointer.new(dtype)
          data_ptr.write_float(value)
        when String
          data_ptr = write_tensor_string([value])
        when TrueClass, FalseClass
          data_ptr = TensorDataPointer.new(:uchar)
          data_ptr.write_uchar(value ? 1 : 0)
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
                                  data_ptr, data_ptr.size,
                                  TensorDataPointer::Deallocator, nil)
      ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
    end

    def value
      value =
        case dtype
        when :float, :double, :int32, :uint8, :int16, :int8, :int64, :uint16, :uint32, :uint64
          data_pointer.send("read_array_of_#{dtype}", element_count)
        when :bfloat16
          byte_str = data_pointer.read_bytes(element_count * 2)
          element_count.times.map { |i| "#{byte_str[(2 * i)..(2 * i + 1)]}\x00\x00".unpack1("g") }
        when :complex64
          data_pointer.read_array_of_float(element_count * 2).each_slice(2).map { |v| Complex(*v) }
        when :complex128
          data_pointer.read_array_of_double(element_count * 2).each_slice(2).map { |v| Complex(*v) }
        when :string
          read_tensor_string(data_pointer)
        when :bool
          data_pointer.read_array_of_int8(element_count).map { |v| v == 1 }
        when :resource, :variant
          return data_pointer
        else
          raise "Unknown type: #{dtype}"
        end

      reshape(value, shape)
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

    def data_pointer
      FFI.TF_TensorData(self)
    end

    def reshape(arr, dims)
      return arr.first if dims.empty?
      arr = arr.flatten
      dims[1..-1].reverse.each do |dim|
        arr = arr.each_slice(dim)
      end
      arr.to_a
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

    # string tensor format
    # https://github.com/tensorflow/tensorflow/blob/5453aee48858fd375172d7ae22fad1557e8557d6/tensorflow/c/tf_tensor.h#L57
    def read_tensor_string(data_pointer)
      start_offset_size = element_count * 8
      offsets = data_pointer.read_array_of_uint64(element_count)
      byte_size = FFI.TF_TensorByteSize(self)
      element_count.times.map do |i|
        str_len = (offsets[i + 1] || (byte_size - start_offset_size)) - offsets[i]
        str = (data_pointer + start_offset_size + offsets[i]).read_bytes(str_len)
        dst = ::FFI::MemoryPointer.new(:char, str.bytesize + 100)
        dst_len = ::FFI::MemoryPointer.new(:size_t)
        Status.check do |status|
          FFI.TF_StringDecode(str, str.bytesize, dst, dst_len, status)
        end
        dst.read_pointer.read_bytes(dst_len.read_int32)
      end
    end

    def write_tensor_string(data)
      start_offset_size = data.size * 8
      offsets = [0]
      data.each do |str|
        offsets << offsets.last + str.bytesize + 1
      end
      data_ptr = TensorDataPointer.new(:char, start_offset_size + offsets.pop)
      data_ptr.write_array_of_uint64(offsets)
      data.zip(offsets) do |str, offset|
        dst_len = FFI.TF_StringEncodedSize(str.bytesize)
        Status.check do |status|
          FFI.TF_StringEncode(str, str.bytesize, data_ptr + start_offset_size + offset, dst_len, status)
        end
      end
      data_ptr
    end
  end
end