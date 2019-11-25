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

        key_outputs = feed_dict.keys.map(&:outputs).flatten
        keys_ptr = FFI::Output.array_to_ptr(key_outputs.map(&:output))

        values = self.values_to_tensors(feed_dict)
        values_ptr = ::FFI::MemoryPointer.new(:pointer, values.length)
        values_ptr.write_array_of_pointer(values)

        # Gather up all the outputs for each operation
        outputs = operations.map do |operation|
          case operation
            when Operation, Variable
              operation.outputs
            when OutputOperation
              operation
            else
              raise(Error::UnimplementedError, "Unsupported operation type: #{operation}")
          end
        end.flatten

        outputs_ptr = FFI::Output.array_to_ptr(outputs.map(&:output))
        results_ptr = ::FFI::MemoryPointer.new(:pointer, outputs.length)

        # Gather up all the targets
        targets = operations.map do |operation|
          case operation
            when Operation, Variable
              operation
            when FFI::Output
              operation.operation(self.graph)
            else
              raise("Unsupported target: #{operation}")
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
                            outputs_ptr, results_ptr, outputs.length,
                            # Targets
                            targets_ptr, operations.length,
                            metadata,
                            status)
        end

        results = results_ptr.read_array_of_pointer(outputs.length).map.with_index do |pointer, i|
          output = outputs[i]
          Tensor.from_pointer(pointer).value
        end

        # For each operation we want to return a single result
        start = 0
        result = operations.reduce(Array.new) do |array, operation|
          length = operation.outputs.length
          if length == 0
            array << nil
          else
            array.concat(results[start, length])
            start += length
          end
          array
        end

        if operations.length == 1
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
              raise(Error::UnknownError, "Cannot determine dtype: #{key}") if key.num_outputs != 1
              dtype = key.output_types.first
              Tensor.new(value, dtype: dtype)
            end
        end
      end
    end
  end
end
