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
    extend FFI::Library
    ffi_lib "#{RbConfig::CONFIG['RUBY_SO_NAME']}.#{RbConfig::CONFIG['SOEXT']}"
    attach_function :ruby_xmalloc, [:size_t], :pointer
    attach_function :ruby_xfree, [:pointer], :void

    attr_reader :count, :size, :type_size

    extend Forwardable
    def_delegators :to_ptr, :+, *::FFI::Pointer.instance_methods.grep(/^write_|^read_/)

    # Store a callback as a class consstant (so it won't be garbage collected ) that tensorflow will trigger
    # when memory should be freed.
    Deallocator = ::FFI::Function.new(:void, [:pointer, :size_t, :pointer]) do |data, len, arg|
                    ruby_xfree(data)
                  end


    def self.write_complex64(value)
      result = TensorData.new(:float, 2)
      result.write_array_of_float([value.real, value.imaginary])
      result
    end

    def self.write_complex128(value)
      result = TensorData.new(:double, 2)
      result.write_array_of_double([value.real, value.imaginary])
      result
    end

    def self.write_string(string)
      self.write_array_of_string(Array(string))
    end

    def self.write_array_of_string(strings)
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
      result = self.new(:char, start_offset_size + encoded_sizes.sum)

      # Write the offsets
      result.write_array_of_uint64(offsets)

      # Write the strings
      strings.each_with_index do |string, index|
        offset = offsets[index]
        size = encoded_sizes[index]
        Status.check do |status|
          FFI.TF_StringEncode(string, string.bytesize, result + start_offset_size + offset, size, status)
        end
      end
      result
    end

    def self.from_pointer(pointer, size, count)
      result = self.allocate
      result.instance_variable_set(:@pointer, pointer)
      result.instance_variable_set(:@size, size)
      result.instance_variable_set(:@count, count)
      result
    end

    def initialize(type, count=1)
        @type_size = ::FFI.type_size(type)
        @count = count
        @size = @type_size * @count
        @pointer = self.class.ruby_xmalloc(@type_size * @count)
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
      offsets = self.read_array_of_uint64(count)

      offsets.map.with_index do |offset, index|
        src_bytes = (offsets[index + 1] || self.size) - offset
        dst_ptr = ::FFI::MemoryPointer.new(:pointer)
        dst_len_ptr = ::FFI::MemoryPointer.new(:size_t)
        Status.check do |status|
          FFI.TF_StringDecode(self + start_offset_size + offset, src_bytes, dst_ptr, dst_len_ptr, status)
        end
        string_pointer = dst_ptr.read_pointer
        string_length = dst_len_ptr.read(:size_t)
        string_pointer.read_string(string_length)
      end
    end
  end
end