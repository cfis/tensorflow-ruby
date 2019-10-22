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
        inputs_ptr, tensors_ptr = self.initialize_inputs(inputs)
        outputs_ptr = self.initialize_outputs(outputs)
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

      def initialize_inputs(inputs)
        inputs_ptr = ::FFI::MemoryPointer.new(FFI::Output, inputs.length)
        tensors_ptr = ::FFI::MemoryPointer.new(:pointer, inputs.length)

        position = 0
        inputs.each do |operation, tensor|
          input = FFI::Output.new(inputs_ptr + position * FFI::Output.size)
          input[:oper] = operation
          input[:index] = 0

          tensor_ptr = (tensors_ptr +  position * FFI::Output.size)
          tensor_ptr.write_pointer(tensor.to_ptr)
          position += 1
        end

        return inputs_ptr, tensors_ptr
      end

      def initialize_outputs(outputs)
        outputs_ptr = ::FFI::MemoryPointer.new(FFI::Output, outputs.length)

        position = 0
        outputs.each do |operation|
          input = FFI::Output.new(outputs_ptr + position * FFI::Output.size)
          input[:oper] = operation
          input[:index] = 0
          position += 1
        end

        outputs_ptr
      end
    end
  end
end