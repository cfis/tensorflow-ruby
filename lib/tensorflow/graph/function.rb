module Tensorflow
  module Graph
    class Function
      attr_reader :output_types, :output_shapes
      def initialize(pointer, output_types, output_shapes)
        @pointer = pointer
        @output_types = output_types
        @output_shapes = output_shapes
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
        Tensorflow::FunctionDef.decode(string)
      ensure
        FFI.TF_DeleteBuffer(buffer)
      end
    end
  end
end
