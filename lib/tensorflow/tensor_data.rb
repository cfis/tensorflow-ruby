require 'rbconfig'
module Tensorflow
  # Tensorflow expects client libraries to allocate memory for the data that a tensor wraps. When a tensor is released,
  # it notifies the client via a callback that gives the client a chance to release the memory.
  #
  # We don't want to use a FFI::MemoryPointer because they are garbage collected. If the underlying data is freed before
  # the tensor is released you get a GC (this can happen even if a Ruby tensor object keeps a reference to the pointer at
  # GC time).
  #
  # Thus this class creates its own memory and fress the memory only after being called bcak by tensorflow.

  class TensorData
    extend ::FFI::Library
    ffi_lib "#{RbConfig::CONFIG['RUBY_SO_NAME']}.#{RbConfig::CONFIG['SOEXT']}"
    attach_function :ruby_xmalloc, [:size_t], :pointer
    attach_function :ruby_xfree, [:pointer], :void

    attr_reader :pointer, :byte_size, :dtype, :shape

    extend Forwardable
    def_delegators :to_ptr, :+, *::FFI::Pointer.instance_methods.grep(/^write_/)

    # Store a callback as a class consstant (so it won't be garbage collected ) that tensorflow will trigger
    # when memory should be freed.
    Deallocator = ::FFI::Function.new(:void, [:pointer, :size_t, :pointer]) do |data, len, arg|
                    ruby_xfree(data)
                  end

    DTYPE_TO_NUMO_TYPE_MAP = {bool:       Numo::Bit,
                              complex64:  Numo::SComplex,
                              complex128: Numo::DComplex,
                              double:     Numo::DFloat,
                              float:      Numo::SFloat,
                              int8:       Numo::Int8,
                              int16:      Numo::Int16,
                              int32:      Numo::Int32,
                              int64:      Numo::Int64,
                              uint8:      Numo::UInt8,
                              uint16:     Numo::UInt16,
                              uint32:     Numo::UInt32,
                              uint64:     Numo::UInt64}

    NUMO_TYPE_TO_DTYPE_MAP = DTYPE_TO_NUMO_TYPE_MAP.each_with_object(Hash.new) do |pair, hash|
      hash[pair.last] = pair.first
    end

    def self.figure_dtype(value)
      case value
        when Numo::RObject
          # Need to look at the first element to see what it is
          self.figure_dtype(value[0])
        when Numo::NArray
          NUMO_TYPE_TO_DTYPE_MAP[value.class]
        when Array
          self.figure_dtype(value.first)
        when Integer
          (value >= -2147483648 && value <= 2147483647) ? :int32 : :int64
        when Complex
          (value.real > -1.175494351e38 && value.real <	3.402823466e38) ? :complex64 : :complex128
        when Numeric
          (value > -1.175494351e38 && value <	3.402823466e38) ? :float : :double
        when String
          :string
        when TrueClass, FalseClass
          :bool
        when ::FFI::Pointer
          :pointer
        when Tensor
          value.dtype
        when Variable
          value.dtype
        when Graph::Operation
          nil
        when Eager::TensorHandle
          value.dtype
        else
          raise(::TensorflowError, "Unsupported type: #{value.class}")
      end
    end

    def self.type_size(dtype)
      case dtype
        when :complex64
          ::FFI.type_size(:float) * 2
        when :complex128
          ::FFI.type_size(:double) * 2
        else
          ::FFI.type_size(dtype)
      end
    end

    def self.value_with_shape(value, dtype, shape)
      if shape && shape.size > 0
        dtype ||= self.figure_dtype(value)
        numo_klass = DTYPE_TO_NUMO_TYPE_MAP[dtype]
        numo_klass.new(shape).fill(value)
      else
        value
      end
    end

    def self.from_pointer(pointer, byte_size, dtype, shape)
      result = self.allocate
      result.instance_variable_set(:@pointer, pointer)
      result.instance_variable_set(:@byte_size, byte_size)
      result.instance_variable_set(:@dtype, dtype)
      result.instance_variable_set(:@shape, shape)
      result
    end

    def initialize(value, dtype: nil, shape: [])
      @dtype = dtype || self.class.figure_dtype(value)
      @shape = shape
      case value
        when Numo::NArray
          self.write_narray(value)
        when Array
          raise(TensorflowError, "TensorData does not support Arrays. Please use a Numo::NArray")
        else
          self.write_scalar(value)
      end
    end

    def to_ptr
      @pointer
    end

    def read_array_of_complex64(count)
      values = self.read_array_of_float(2 * count)
      values.each_slice(2).map do |real, imaginary|
        Complex(real, imaginary)
      end
    end

    def read_array_of_complex128(count)
      values = self.read_array_of_double(2 * count)
      values.each_slice(2).map do |real, imaginary|
        Complex(real, imaginary)
      end
    end

    def read_array_of_string(count)
      # The start of the data section comes after the offset table
      start_offset_size = count * ::FFI.type_size(:int64)

      # Read in the string offsets
      offsets = self.pointer.read_array_of_uint64(count)

      offsets.map.with_index do |offset, index|
        src_bytes = (offsets[index + 1] || self.byte_size) - offset
        dst_ptr = ::FFI::MemoryPointer.new(:pointer)
        dst_len_ptr = ::FFI::MemoryPointer.new(:size_t)
        Status.check do |status|
          FFI.TF_StringDecode(self.pointer + start_offset_size + offset, src_bytes, dst_ptr, dst_len_ptr, status)
        end
        string_pointer = dst_ptr.read_pointer
        string_length = dst_len_ptr.read(:size_t)
        string_pointer.read_string(string_length)
      end
    end

    # def value
    #   result = case self.dtype
    #             when :float, :double, :int32, :uint8, :int16, :int8, :int64, :uint16, :uint32, :uint64
    #               self.pointer.send("read_array_of_#{self.dtype}", self.count)
    #             when :bfloat16
    #               byte_str = self.pointer.read_bytes(self.count * 2)
    #               self.count.times.map { |i| "#{byte_str[(2 * i)..(2 * i + 1)]}\x00\x00".unpack1("g") }
    #             when :complex64
    #               self.read_array_of_complex64(self.count)
    #             when :complex128
    #               self.read_array_of_complex128(self.count)
    #             when :string
    #               self.read_array_of_string(self.count)
    #             when :bool
    #               self.pointer.read_array_of_int8(self.count)
    #             when :resource, :variant
    #               return self.data
    #             else
    #               raise "Unsupported tensor data type: #{self.dtype}"
    #           end
    #
    #   if self.count == 1
    #     result.first
    #   else
    #     result
    #   end
    # end

    def value
      result = case self.dtype
                 when :bfloat16
                   byte_str = self.pointer.read_bytes(self.count * 2)
                   self.count.times.map { |i| "#{byte_str[(2 * i)..(2 * i + 1)]}\x00\x00".unpack1("g") }
                 when :string
                   count = self.shape.reduce(1) {|dim, result| result *= dim}
                   self.read_array_of_string(count)
                 when :bool
                   bytes = self.pointer.read_bytes(self.byte_size)
                   int8 = if self.shape.empty?
                             Numo::Int8.from_binary(bytes)
                           else
                             Numo::Int8.from_binary(bytes, self.shape)
                           end
                   int8.cast_to(Numo::Bit)
                 else
                   bytes = self.pointer.read_bytes(self.byte_size)
                   numo_klass = DTYPE_TO_NUMO_TYPE_MAP[self.dtype]
                   if self.shape.empty?
                     numo_klass.from_binary(bytes)
                   else
                     numo_klass.from_binary(bytes, self.shape)
                   end
               end

      if self.shape.empty?
        result[0]
      else
        result
      end
    end

    def write_array_of_string(strings)
      # The start of the data section comes after the offset table
      start_offset_size = strings.size * ::FFI.type_size(:int64)

      # Get the encoded sizes for each string
      encoded_sizes = strings.map do |string|
        FFI.TF_StringEncodedSize(string.bytesize)
      end

      # Now figure the offsets. Offsets are relative to the beginning of data section, not the beginning of the pointer.
      # Notice we skip the last string [0..-2] since its offset would be the end of the pointer
      offsets = [0]
      encoded_sizes[0..-2].each do |encoded_size|
        offsets << offsets.last + encoded_size
      end

      # Allocate the needed memory
      @byte_size = start_offset_size + encoded_sizes.sum
      @pointer = self.class.ruby_xmalloc(@byte_size)

      # Write the offsets
      self.pointer.write_array_of_uint64(offsets)

      # Write the strings
      strings.each_with_index do |string, index|
        offset = offsets[index]
        size = encoded_sizes[index]
        Status.check do |status|
          FFI.TF_StringEncode(string, string.bytesize, self.pointer + start_offset_size + offset, size, status)
        end
      end
    end

    def write_narray(new_value)
      @shape = new_value.shape

      case self.dtype
        when :string
          self.write_array_of_string(new_value.flatten.to_a)
        when :bool
          new_value = new_value.cast_to(Numo::Int8)
          @byte_size = new_value.byte_size
          @pointer = self.class.ruby_xmalloc(self.byte_size)
          self.pointer.write_bytes(new_value.to_binary)
        else
          if NUMO_TYPE_TO_DTYPE_MAP[new_value.class] != dtype
            # Cast the narray if necessary (say the user passed in float but we have a double array)
            new_value = new_value.cast_to(DTYPE_TO_NUMO_TYPE_MAP[dtype])
          end
          @byte_size = new_value.byte_size
          @pointer = self.class.ruby_xmalloc(@byte_size)
          self.pointer.write_bytes(new_value.to_binary)
      end
    end

    def write_scalar(new_value)
      case self.dtype
        when :string
          self.write_array_of_string([new_value])
        when :resource, :variant
          return self
        else
          @byte_size = self.class.type_size(self.dtype)
          @pointer = self.class.ruby_xmalloc(@byte_size)
          case self.dtype
            when :float, :double, :int32, :uint8, :int16, :int8, :int64, :uint16, :uint32, :uint64
              self.pointer.send("write_#{self.dtype}", new_value)
            when :bfloat16
              byte_str = self.pointer.read_bytes(self.count * 2)
              self.count.times.map { |i| "#{byte_str[(2 * i)..(2 * i + 1)]}\x00\x00".unpack1("g") }
            when :complex64
              self.pointer.write_array_of_float([new_value.real, new_value.imaginary])
            when :complex128
              self.pointer.write_array_of_double([new_value.real, new_value.imaginary])
            when :bool
              self.pointer.write_int8(new_value ? 1 : 0)
            else
              raise "Unsupported tensor data type: #{self.dtype}"
          end
      end
    end
  end
end