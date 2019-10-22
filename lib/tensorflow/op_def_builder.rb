module Tensorflow
  class OpDefBuilder
    def self.unknown_shape_inference_func
      @unknown_shape_func ||= FFI.ffi_libraries.first.find_function('TF_ShapeInferenceContextSetUnknownShape')
    end

    def self.finalize(pointer)
      proc do
        FFI::TF_DeleteOpDefinitionBuilder(pointer)
      end
    end

    def initialize(name)
      @pointer = FFI.TF_NewOpDefinitionBuilder(name)
      ObjectSpace.define_finalizer(self, self.class.finalize(@pointer))
    end

    def to_ptr
      @pointer
    end

    def attr(spec)
      FFI.TF_OpDefinitionBuilderAddAttr(self, spec)
      self
    end

    def input(spec)
      FFI.TF_OpDefinitionBuilderAddInput(self, spec)
      self
    end

    def output(spec)
      FFI.TF_OpDefinitionBuilderAddOutput(self, spec)
      self
    end

    def shape_inference(func)
      FFI.TF_OpDefinitionBuilderSetShapeInferenceFunction(self, func)
      self
    end

    def register
      ObjectSpace.undefine_finalizer(self)
      Status.check do |status|
        FFI.TF_RegisterOpDefinition(self, status)
      end
    end
  end
end