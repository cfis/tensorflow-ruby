module Tensorflow
  module Graph
    class Function
      def initialize(pointer)
        @pointer = pointer
      end

      def to_ptr
        @pointer
      end

      def name
        name, ptr = FFI.TF_FunctionName(self)
        name
      end

      def function_def
        buffer_ptr = FFI.TF_NewBuffer
        Status.check do |status|
          FFI.TF_FunctionToFunctionDef(self, buffer_ptr, status)
        end
        buffer = FFI::Buffer.new(buffer_ptr)
        string = buffer[:data].read_string(buffer[:length])
        FunctionDef.decode(string)
      ensure
        FFI.TF_DeleteBuffer(buffer)
      end
    end
  end
end
