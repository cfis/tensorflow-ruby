module Tensorflow
  module Eager
    class Context
      def self.default
        @default ||= Context.new
      end

      def self.finalize(pointer)
        proc { FFI.TFE_DeleteContext(pointer) }
      end

      def initialize
        options = FFI.TFE_NewContextOptions
        Status.check do |status|
          @pointer = FFI.TFE_NewContext(options, status)
        end
        ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
        FFI.TFE_DeleteContextOptions(options)
      end

      def execute(op_name, inputs = [], **attrs)
        operation = Operation.new(self)
        operation.execute(op_name, inputs, **attrs)
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
