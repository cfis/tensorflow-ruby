module Tensorflow
  module Graph
    class GraphDefOptions
      def self.finalize(pointer)
        proc do
          FFI.TF_DeleteImportGraphDefOptions(pointer)
        end
      end

      def initialize
        @pointer = FFI.TF_NewImportGraphDefOptions()
        ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
      end

      def to_ptr
        @pointer
      end

      def prefix=(value)
        FFI.TF_ImportGraphDefOptionsSetPrefix(self, value)
      end
    end
  end
end