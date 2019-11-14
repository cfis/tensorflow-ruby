module Tensorflow
  module Eager
    class Context
      extend Forwardable
      def_delegators :@name_scope, :name_scope, :scoped_name, :unique_name

      def self.default
        @default ||= Context.new
      end

      def self.finalize(pointer)
        proc { FFI.TFE_DeleteContext(pointer) }
      end

      def initialize
        @name_scope = NameScope.new
        options = FFI.TFE_NewContextOptions
        Status.check do |status|
          @pointer = FFI.TFE_NewContext(options, status)
        end
        ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
        FFI.TFE_DeleteContextOptions(options)
      end

      def as_default
        raise(TensorflowError, "Must provide block") unless block_given?
        ExecutionContext.push(self)
        begin
          yield self
        ensure
          ExecutionContext.pop
        end
      end

      def create_operation(op_type, inputs=[], attrs={})
        Operation.new(self, op_type, inputs, attrs)
      end

      def execute(operation)
        # TODO decide how many retvals to allocate
        retvals = ::FFI::MemoryPointer.new(:pointer, 10)
        num_retvals = ::FFI::MemoryPointer.new(:int)
        num_retvals.write_int(retvals.size)

        Status.check do |status|
          FFI.TFE_Execute(operation, retvals, num_retvals, status)
        end

        n = num_retvals.read_int
        if n > 0
          handles = retvals.read_array_of_pointer(n).map do |handle|
            TensorHandle.new(self, handle)
          end

          # TODO handle case where n = 1 and still want an array for retvals
          n == 1 ? handles.first : handles
        end
      end

      def device_policy
        FFI::ContextDevicePlacementPolicy[FFI.TFE_ContextGetDevicePlacementPolicy(@pointer)]
      end

      def enable_run_metadata
        FFI.TFE_ContextEnableRunMetadata(@pointer)
      end

      def disable_run_metadata
        FFI.TFE_ContextDisableRunMetadata(@pointer)
      end

      def start_step
        FFI.TFE_ContextStartStep(@pointer)
      end

      def end_step
        FFI.TFE_ContextEndStep(@pointer)
      end

      def to_ptr
        @pointer
      end

      def shared_name
        # hard-coded in Python library
        "cd2c89b7-88b7-44c8-ad83-06c2a9158347"
      end

      def add_function(function)
        Status.check do |status|
          FFI.TFE_ContextAddFunction(self, function, status)
        end
      end

      def remove_function(function)
        name = function.is_a?(Graph::Function) ? function.name : function
        Status.check do |status|
          FFI.TFE_ContextRemoveFunction(self, name, status)
        end
      end

      def function?(function)
        name = function.is_a?(Graph::Function) ? function.name : function
        # result is uchar
        FFI.TFE_ContextHasFunction(self, name) != 0
      end
    end
  end
end
