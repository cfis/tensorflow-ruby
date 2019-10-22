module Tensorflow
  module Graph
    class SessionOptions
      attr_accessor :target
      def self.finalize(pointer)
        proc do
          FFI.TF_DeleteSessionOptions(pointer)
        end
      end

      def initialize
        @pointer = FFI.TF_NewSessionOptions
        ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
      end

      def to_ptr
        @pointer
      end
    end

    class Session
      attr_accessor :options

      def self.finalize(pointer)
        proc do
          FFI.TF_DeleteSession(pointer)
        end
      end

      def initialize(graph, options)
        Status.check do |status|
          @pointer = FFI.TF_NewSession(graph, options, status)
        end
      end

      def to_ptr
        @pointer
      end

      def run(inputs, outputs)
        operations = Array.new
        tensors = Array.new
        inputs.each do |operation, tensor|
          operations << operation
          tensors << tensor
        end

        inputs_ptr = FFI::Output.pointer_array(operations)
        tensors_ptr = self.initialize_tensors(tensors)
        outputs_ptr = FFI::Output.pointer_array(outputs)
        #targets = self.initialize_targets(targets)
        result_ptr = ::FFI::MemoryPointer.new(:pointer, outputs.length)

        run_options = nil
        targets = nil
        metadata = nil

        Status.check do |status|
          FFI.TF_SessionRun(self, run_options,
                            inputs_ptr, tensors_ptr, inputs.length,
                            outputs_ptr, result_ptr, outputs.length,
                            targets, 0,
                            metadata,
                            status)
        end

        result_ptr.read_array_of_pointer(outputs.length).map do |pointer|
          Tensor.new(:pointer => pointer)
        end
      end

      def close
        Status.check do |status|
          FFI.TF_CloseSession(self, status)
        end
      end

      def initialize_tensors(tensors)
        tensors_ptr = ::FFI::MemoryPointer.new(:pointer, tensors.length)

        tensors.each_with_index do |tensor, index|
          tensor_ptr = (tensors_ptr +  index * FFI::Output.size)
          tensor_ptr.write_pointer(tensor.to_ptr)
        end

        tensors_ptr
      end
    end
  end
end
