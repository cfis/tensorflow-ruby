# Status holds error information returned by TensorFlow. We
# can use status to get error and even display the error messages from tensorflow.
module Tensorflow
  class Status
    def self.finalize(pointer)
      proc do
        FFI::TF_DeleteStatus(pointer)
      end
    end

    def self.check
      status = Status.new
      result = yield status
      status.check
      status = nil
      result
    end

    def initialize
      @pointer = FFI.TF_NewStatus
      ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
    end

    def to_ptr
      @pointer
    end

    def code
      FFI.TF_GetCode(self)
    end

    def message
      FFI.TF_Message(self)
    end

    def set(code, message)
      FFI.TF_SetStatus(self, code, message)
    end

    def check
      if self.code != :tf_ok
        raise(::TensorflowError, self.message)
      end
    end
  end
end