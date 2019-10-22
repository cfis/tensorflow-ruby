require 'rbconfig'
module Tensorflow
  # Tensorflow expects client libraries to allocate memory for the data that a tensor wraps. When a tensor is released,
  # it calls back to the client via callback to give the client a chance to release the memory.
  #
  # By default when an FFI MemoryPointer is garbage collected, it frees its assoicated memory, resulting in crashes. Even if
  # a Ruby tensor object keeps a reference to the pointer, the interplay between when the ruby oject is collected (thus removing
  # the reference to the pointer), when the ruby pointer is collected and when the c object calls back to Ruby results in
  # fairly frequent crashes.
  #
  # So instead we create a new Pointer class that does *not* garbage collect its memory. Instead the memory is only freed when
  # the callback is triggered from tensorflow. Note that the FFI MemoryPointer allocates memory using ruby_xmalloc function from
  # the ruby runtime library. Thus the memory *must* be freed using ruby_xfree.
  #
  # An alternative implementation would be directly using malloc/free from the C library, but that turns out to require a bit
  # more wrapping code (need to keep track of the pointer length), and thus this solution was chosen instead.
  class TensorDataPointer < ::FFI::MemoryPointer
    # Get access to Ruby's ruby_xfree function
    extend FFI::Library
    ffi_lib "#{RbConfig::CONFIG['RUBY_SO_NAME']}.#{RbConfig::CONFIG['SOEXT']}"
    attach_function :ruby_xfree, [:pointer], :void

    # Store a callback as a class consstant (so it won't be garbage collected ) that tensorflow will trigger
    # when memory should be freed.
    Deallocator = ::FFI::Function.new(:void, [:pointer, :size_t, :pointer]) do |data, len, arg|
                    ruby_xfree(data)
                  end

    def initialize(type, count=1)
      super(type, count)
      # Tell ruby to not garbage collect this memory!
      self.autorelease = false
    end
  end
end