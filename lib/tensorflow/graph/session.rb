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
      attr_accessor :graph, :options

      def self.run(graph)
        session = self.new(graph, SessionOptions.new)
        result = yield session
        session.close
        result
      end

      def self.finalize(pointer)
        proc do
          FFI.TF_DeleteSession(pointer)
        end
      end

      def initialize(graph, options)
        @graph = graph
        Status.check do |status|
          @pointer = FFI.TF_NewSession(graph, options, status)
        end
      end

      def to_ptr
        @pointer
      end

      def run(operations, feed_dict={})
        operations = Array(operations).flatten.compact

        key_outputs = feed_dict.keys.map(&:to_output)
        keys_ptr = FFI::Output.array_to_ptr(key_outputs)

        values = self.values_to_tensors(feed_dict)
        values_ptr = ::FFI::MemoryPointer.new(:pointer, values.length)
        values_ptr.write_array_of_pointer(values)

        # Gather up all the outputs for each operation
        outputs = operations.map do |operation|
          case operation
            when Operation
              operation.outputs
            when FFI::Output
              operation
          end
        end.flatten

        outputs_ptr = FFI::Output.array_to_ptr(outputs)
        result_ptr = ::FFI::MemoryPointer.new(:pointer, outputs.length)

        # Gather up all the targets
        targets = operations.map do |operation|
          case operation
            when Operation
              operation
            when FFI::Output
              operation.operation(self.graph)
          end
        end
        targets_ptr = ::FFI::MemoryPointer.new(:pointer, targets.length)
        targets_ptr.write_array_of_pointer(targets)

        run_options = nil
        metadata = nil

        Status.check do |status|
          FFI.TF_SessionRun(self, run_options,
                            # Inputs
                            keys_ptr, values_ptr, feed_dict.keys.length,
                            # Outputs
                            outputs_ptr, result_ptr, outputs.length,
                            # Targets
                            targets_ptr, operations.length,
                            metadata,
                            status)
        end

        result = result_ptr.read_array_of_pointer(outputs.length).map.with_index do |pointer, i|
          output = outputs[i]
          Tensor.from_pointer(pointer).value
        end

        if outputs.length == 1
          result.first
        else
          result
        end
      end

      def close
        Status.check do |status|
          FFI.TF_CloseSession(self, status)
        end
      end

      def values_to_tensors(values)
        values.map do |key, value|
          case value
            when Tensor
              value
            else
              # The value dtype needs to match the key dtype
              raise(TensorflowError, "Cannot determine dtype: #{key}") if key.num_outputs != 1
              dtype = key.output_types.first
              Tensor.new(value, dtype: dtype)
            end
        end
      end
    end
  end
end
