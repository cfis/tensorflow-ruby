module Tensorflow
  class Tensor
    include TensorMixin
    def self.finalize(pointer)
      proc do
        FFI.TF_DeleteTensor(pointer)
      end
    end

    def initialize(value = nil, dtype: nil, shape: nil, pointer: nil)
      # Pointer should be a tensor handle, not a tensor
      if pointer
        @pointer = pointer
      else
        data = value
        data = Array(data) unless data.is_a?(Array) || data.is_a?(Numo::NArray)
        shape ||= calculate_shape(value)

        if shape.size > 0
          dims_ptr = TensorDataPointer.new(:int64, shape.size)
          dims_ptr.write_array_of_int64(shape)
        else
          dims_ptr = nil
        end

        if data.is_a?(Numo::NArray)
          dtype ||= Utils.infer_type(data)
          # TODO use Numo read pointer?
          data_ptr = TensorDataPointer.new(:uchar, data.byte_size)
          data_ptr.write_bytes(data.to_string)
        else
          data = data.flatten
          dtype ||= Utils.infer_type(data)
          case dtype
          when :float, :double, :int32, :uint8, :int16, :int8, :int64, :uint16, :uint32, :uint64
            data_ptr = TensorDataPointer.new(dtype, data.size)
            data_ptr.send("write_array_of_#{dtype}", data)
          when :bfloat16
            # https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
            data_ptr = TensorDataPointer.new(:int8, data.size * 2)
            data_ptr.write_bytes(data.map { |v| [v].pack("g")[0..1] }.join)
          when :complex64
            data_ptr = TensorDataPointer.new(:float, data.size * 2)
            data_ptr.write_array_of_float(data.flat_map { |v| [v.real, v.imaginary] })
          when :complex128
            data_ptr = TensorDataPointer.new(:double, data.size * 2)
            data_ptr.write_array_of_double(data.flat_map { |v| [v.real, v.imaginary] })
          when :string
            data_ptr = string_ptr(data)
          when :bool
            data_ptr = TensorDataPointer.new(:int8, data.size)
            data_ptr.write_array_of_int8(data.map { |v| v ? 1 : 0 })
          else
            raise "Unknown type: #{dtype}"
          end
        end

        # callback = ::FFI::Function.new(:void, [:pointer, :size_t, :pointer]) do |data, len, arg|
        #   TensorData::Deallocator
        # end
        @data_ptr = data_ptr
        type = FFI::DataType[dtype]
        @pointer = FFI.TF_NewTensor(type, dims_ptr, shape.size, data_ptr, data_ptr.size, TensorDataPointer::Deallocator, nil)
      end

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
          # string tensor format
          # https://github.com/tensorflow/tensorflow/blob/5453aee48858fd375172d7ae22fad1557e8557d6/tensorflow/c/tf_tensor.h#L57
          start_offset_size = element_count * 8
          offsets = data_pointer.read_array_of_uint64(element_count)
          byte_size = FFI.TF_TensorByteSize(self)
          element_count.times.map do |i|
            str_len = (offsets[i + 1] || (byte_size - start_offset_size)) - offsets[i]
            str = (data_pointer + start_offset_size + offsets[i]).read_bytes(str_len)
            dst = TensorDataPointer.new(:char, str.bytesize + 100)
            dst_len = TensorDataPointer.new(:size_t)
            Status.check do |status|
              FFI.TF_StringDecode(str, str.bytesize, dst, dst_len, status)
            end
            dst.read_pointer.read_bytes(dst_len.read_int32)
          end
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
    def string_ptr(data)
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