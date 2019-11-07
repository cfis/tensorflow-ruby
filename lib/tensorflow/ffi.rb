module Tensorflow
  module FFI
    extend ::FFI::Library

    begin
      ffi_lib Tensorflow.ffi_lib
    rescue LoadError => e
      raise e if ENV["TENSORFLOW_DEBUG"]
      raise LoadError, "Could not find Tensorflow"
    end

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_attrtype.h
    AttrType = enum(:string, :int, :float, :bool, :type, :shape, :tensor, :placeholder, :func)

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_datatype.h
    DataType = enum(:float, 1, :double, :int32, :uint8, :int16, :int8, :string, :complex64, :int64, :bool, :qint8, :quint8, :qint32, :bfloat16, :qint16, :quint16, :uint16, :complex128, :half, :resource, :variant, :uint32, :uint64)

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h
    class Buffer < ::FFI::Struct
      layout :data, :pointer,
        :length, :size_t,
        :data_deallocator, :pointer
    end
    attach_function :TF_NewBuffer, [], :pointer
    attach_function :TF_DeleteBuffer, [:pointer], :void
    attach_function :TF_GetBuffer, [:pointer], :pointer

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h
    attach_function :TF_Version, [], :string
    attach_function :TF_GetAllOpList, [], Buffer.by_ref

    class Input < ::FFI::Struct
      layout :oper, :pointer,
             :index, :int

      def self.pointer_array(operations)
        result = ::FFI::MemoryPointer.new(self, operations.length)
        operations.each_with_index do |operation, i|
          input = self.new(result + (i * self.size))
          input[:oper] = operation
          input[:index] = 0
        end
        result
      end
    end

    class Output < ::FFI::Struct
      layout :oper, :pointer,
             :index, :int

      def self.array_to_ptr(outputs)
        result = ::FFI::MemoryPointer.new(self, outputs.length)
        outputs.each_with_index do |output, i|
          copy_output = self.new(result[i])
          copy_output[:oper] = output[:oper]
          copy_output[:index] = output[:index]
        end
        result
      end

      def operation(graph)
        Graph::Operation.new(graph, self[:oper])
      end

      def to_s
        "#{self[:oper]}: #{self[:index]}"
      end
    end

    attach_function :TF_NewGraph, [], :pointer
    attach_function :TF_DeleteGraph, [:pointer], :pointer
    attach_function :TF_GraphGetOpDef, [:pointer, :string, :pointer, :pointer], :void
    attach_function :TF_GraphGetTensorNumDims, [:pointer, Output, :pointer], :int

    attach_function :TF_GraphGetTensorShape, [:pointer, Output, :pointer, :int, :pointer], :void
    attach_function :TF_GraphSetTensorShape, [:pointer, Output, :pointer, :int, :pointer], :void

    attach_function :TF_NewOperation, [:pointer, :string, :string], :pointer
    attach_function :TF_FinishOperation, [:pointer, :pointer], :pointer
    attach_function :TF_SetDevice, [:pointer, :string,], :void
    attach_function :TF_SetAttrBool, [:pointer, :string, :uchar], :void
    attach_function :TF_SetAttrBoolList, [:pointer, :string, :pointer, :int], :void
    attach_function :TF_SetAttrInt, [:pointer, :string, :int64], :void
    attach_function :TF_SetAttrIntList, [:pointer, :string, :pointer, :int], :void
    attach_function :TF_SetAttrFloat, [:pointer, :string, :float], :void
    attach_function :TF_SetAttrFloatList, [:pointer, :string, :pointer, :int], :void
    attach_function :TF_SetAttrFuncName, [:pointer, :string, :string, :size_t], :void
    attach_function :TF_SetAttrPlaceholder, [:pointer, :string, :string], :void
    attach_function :TF_SetAttrShape, [:pointer, :string, :pointer, :int], :void
    attach_function :TF_SetAttrShapeList, [:pointer, :string, :pointer, :pointer, :int], :void
    attach_function :TF_SetAttrString, [:pointer, :string, :pointer, :size_t], :void
    attach_function :TF_SetAttrStringList, [:pointer, :string, :pointer, :pointer, :int], :void
    attach_function :TF_SetAttrType, [:pointer, :string, DataType], :void
    attach_function :TF_SetAttrTypeList, [:pointer, :string, :pointer, :int], :void
    attach_function :TF_SetAttrTensor, [:pointer, :string, :pointer, :pointer], :void
    attach_function :TF_SetAttrTensorList, [:pointer, :string, :pointer, :int, :pointer], :void

    attach_function :TF_AddInput, [:pointer, Output], :void
    attach_function :TF_AddInputList, [:pointer, :pointer, :int], :void

    attach_function :TF_AddControlInput, [:pointer, :pointer], :void
    attach_function :TF_OperationNumControlInputs, [:pointer], :int
    attach_function :TF_OperationGetControlInputs, [:pointer, :pointer, :int], :int
    attach_function :TF_OperationNumControlOutputs, [:pointer], :int
    attach_function :TF_OperationGetControlOutputs, [:pointer, :pointer, :int], :int

    attach_function :TF_OperationToNodeDef, [:pointer, :pointer, :pointer], :void
    attach_function :TF_OperationNumInputs, [:pointer], :int
    attach_function :TF_OperationInputType, [Input], DataType
    attach_function :TF_OperationInputListLength, [:pointer, :string, :pointer], :int
    attach_function :TF_OperationAllInputs, [:pointer, :pointer, :int], :void

    attach_function :TF_OperationNumOutputs, [:pointer], :int
    attach_function :TF_OperationOutputType, [Output], DataType
    attach_function :TF_OperationOutputListLength, [:pointer, :string, :pointer], :int

    attach_function :TF_OperationOutputNumConsumers, [Output], :int
    attach_function :TF_OperationOutputConsumers, [Output, :pointer, :int], :int

    class AttrMetadata < ::FFI::Struct
      layout :is_list, :uchar,
             :list_size, :int64,
             :type, AttrType,
             :total_size, :int64
    end

    attach_function :TF_OperationGetAttrMetadata, [:pointer, :string, :pointer], AttrMetadata.by_value
    attach_function :TF_OperationGetAttrBool, [:pointer, :string, :pointer, :pointer], :void
    attach_function :TF_OperationGetAttrBoolList, [:pointer, :string, :pointer, :int, :pointer], :void
    attach_function :TF_OperationGetAttrFloat, [:pointer, :string, :pointer, :pointer], :void
    attach_function :TF_OperationGetAttrFloatList, [:pointer, :string, :pointer, :int, :pointer], :void
    attach_function :TF_OperationGetAttrInt, [:pointer, :string, :pointer, :pointer], :void
    attach_function :TF_OperationGetAttrIntList, [:pointer, :string, :pointer, :int, :pointer], :void
    attach_function :TF_OperationGetAttrShape, [:pointer, :string, :pointer, :int, :pointer], :void
    attach_function :TF_OperationGetAttrShapeList, [:pointer, :string, :pointer, :pointer, :int, :pointer, :int, :pointer], :void
    attach_function :TF_OperationGetAttrString, [:pointer, :string, :pointer, :size_t, :pointer], :void
    attach_function :TF_OperationGetAttrStringList, [:pointer, :string, :pointer, :pointer, :int, :pointer, :size_t], :void
    attach_function :TF_OperationGetAttrTensor, [:pointer, :string, :pointer, :pointer], :void
    attach_function :TF_OperationGetAttrType, [:pointer, :string, :pointer, :pointer], :void
    attach_function :TF_OperationGetAttrTypeList, [:pointer, :string, :pointer, :int, :pointer], :void

    attach_function :TF_GraphOperationByName, [:pointer, :string], :pointer
    attach_function :TF_GraphNextOperation, [:pointer, :pointer], :pointer

    attach_function :TF_OperationName, [:pointer], :string
    attach_function :TF_OperationOpType, [:pointer], :string
    attach_function :TF_OperationDevice, [:pointer], :string

    attach_function :TF_AddGradients, [:pointer, :pointer, :int, :pointer, :int, :pointer, :pointer, :pointer], :void
    attach_function :TF_AddGradientsWithPrefix, [:pointer, :string, :pointer, :int, :pointer, :int, :pointer, :pointer, :pointer], :void

    attach_function :TF_NewSessionOptions, [], :pointer
    attach_function :TF_SetTarget, [:pointer, :string], :void
    attach_function :TF_SetConfig, [:pointer, :pointer, :size_t, :pointer], :void
    attach_function :TF_DeleteSessionOptions, [:pointer,], :void

    attach_function :TF_NewSession, [:pointer, :pointer, :pointer], :pointer
    attach_function :TF_CloseSession, [:pointer, :pointer], :void
    attach_function :TF_DeleteSession, [:pointer, :pointer], :void
    attach_function :TF_SessionRun, [:pointer, Buffer,
                                     :pointer, :pointer, :int,
                                     :pointer, :pointer, :int,
                                     :pointer, :int,
                                     Buffer,
                                     :pointer], :void

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_status.h
    StatusCode = enum(:tf_ok, 0,
                      :tf_cancelled, 1,
                      :tf_unknown, 2,
                      :tf_invalid_argument, 3,
                      :tf_deadline_exceeded, 4,
                      :tf_not_found, 5,
                      :tf_already_exists, 6,
                      :tf_permission_denied, 7,
                      :tf_unauthenticated, 16,
                      :tf_resource_exhausted, 8,
                      :tf_failed_precondtion, 9,
                      :tf_aborted, 10,
                      :tf_out_of_range, 11,
                      :tf_unimplemented, 12,
                      :tf_internal, 13,
                      :tf_unavailable, 14,
                      :tf_data_loss, 15)

    attach_function :TF_NewStatus, [], :pointer
    attach_function :TF_DeleteStatus, [:pointer], :pointer
    attach_function :TF_GetCode, [:pointer], StatusCode
    attach_function :TF_Message, [:pointer], :string
    attach_function :TF_SetStatus, [:pointer, StatusCode, :string], :void

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_tensor.h
    callback :tensor_deallocator, [:pointer, :size_t, :pointer], :void
    attach_function :TF_NewTensor, [:int, :pointer, :int, :pointer, :size_t, :tensor_deallocator, :pointer], :pointer
    attach_function :TF_DeleteTensor, [:pointer], :void
    attach_function :TF_TensorType, [:pointer], DataType
    attach_function :TF_NumDims, [:pointer], :int
    attach_function :TF_Dim, [:pointer, :int], :int64
    attach_function :TF_TensorByteSize, [:pointer], :size_t
    attach_function :TF_TensorElementCount, [:pointer], :int64
    attach_function :TF_TensorData, [:pointer], :pointer
    attach_function :TF_TensorByteSize, [:pointer], :size_t
    attach_function :TF_StringEncode, [:pointer, :size_t, :pointer, :size_t, :pointer], :size_t
    attach_function :TF_StringDecode, [:pointer, :size_t, :pointer, :pointer, :pointer], :size_t
    attach_function :TF_StringEncodedSize, [:size_t], :size_t

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/eager/c_api.h
    ContextDevicePlacementPolicy = enum(:explicit, :warn, :silent, :silent_for_int32)

    attach_function :TFE_NewContextOptions, [], :pointer
    attach_function :TFE_ContextOptionsSetAsync, [:pointer, :char], :void
    attach_function :TFE_DeleteContextOptions, [:pointer], :void
    attach_function :TFE_NewContext, [:pointer, :pointer], :pointer
    attach_function :TFE_DeleteContext, [:pointer], :void
    attach_function :TFE_ContextListDevices, [:pointer, :pointer], :pointer
    attach_function :TFE_ContextGetDevicePlacementPolicy, [:pointer], :int
    attach_function :TFE_ContextAddFunction, [:pointer, :pointer, :pointer], :void
    attach_function :TFE_ContextHasFunction, [:pointer, :string], :uchar
    attach_function :TFE_ContextRemoveFunction, [:pointer, :string, :pointer], :void
    attach_function :TFE_ContextAddFunction, [:pointer, :pointer, :pointer], :void
    attach_function :TFE_ContextHasFunction, [:pointer, :string], :uchar
    attach_function :TFE_ContextRemoveFunction, [:pointer, :string, :pointer], :void

    attach_function :TFE_NewTensorHandle, [:pointer, :pointer], :pointer
    attach_function :TFE_DeleteTensorHandle, [:pointer], :void
    attach_function :TFE_TensorHandleDataType, [:pointer], DataType
    attach_function :TFE_TensorHandleNumDims, [:pointer, :pointer], :int
    attach_function :TFE_TensorHandleNumElements, [:pointer, :pointer], :int64
    attach_function :TFE_TensorHandleDim, %i[pointer int pointer], :int64
    attach_function :TFE_TensorHandleDeviceName, [:pointer, :pointer], :string
    attach_function :TFE_TensorHandleBackingDeviceName, [:pointer, :pointer], :string
    attach_function :TFE_TensorHandleResolve, [:pointer, :pointer], :pointer
    attach_function :TFE_NewOp, [:pointer, :string, :pointer], :pointer
    attach_function :TFE_DeleteOp, [:pointer], :void
    attach_function :TFE_OpSetDevice, [:pointer, :string, :pointer], :pointer
    attach_function :TFE_OpGetDevice, [:pointer, :pointer], :string
    attach_function :TFE_OpAddInput, [:pointer, :pointer, :pointer], :void
    attach_function :TFE_OpAddInputList, %i[pointer pointer int pointer], :void
    attach_function :TFE_OpGetAttrType, %i[pointer string pointer pointer], AttrType
    attach_function :TFE_OpSetAttrString, %i[pointer string pointer size_t], :void
    attach_function :TFE_OpSetAttrInt, %i[pointer string int64_t], :void
    attach_function :TFE_OpSetAttrFloat, %i[pointer string float], :void
    attach_function :TFE_OpSetAttrFunction, [:pointer, :string, :pointer], :void
    attach_function :TFE_OpSetAttrFunctionName, [:pointer, :string, :string, :size_t], :void
    attach_function :TFE_OpSetAttrFunctionList, [:pointer, :string, :pointer, :int], :void
    attach_function :TFE_OpSetAttrBool, %i[pointer string uint8], :void
    attach_function :TFE_OpSetAttrTensor, %i[pointer string pointer pointer], :void
    attach_function :TFE_OpSetAttrType, %i[pointer string int], :void
    attach_function :TFE_OpSetAttrShape, %i[pointer string pointer int pointer], :void
    attach_function :TFE_OpSetAttrIntList, %i[pointer string pointer int], :void
    attach_function :TFE_OpSetAttrFloatList, %i[pointer string pointer int], :void
    attach_function :TFE_OpSetAttrTypeList, %i[pointer string pointer int], :void
    attach_function :TFE_OpSetAttrShapeList, %i[pointer string pointer pointer int pointer], :void
    attach_function :TFE_Execute, %i[pointer pointer pointer pointer], :pointer
    attach_function :TFE_ContextHasFunction, [:pointer, :string], :uchar
    attach_function :TFE_ContextEnableRunMetadata, [:pointer], :void
    attach_function :TFE_ContextDisableRunMetadata, [:pointer], :void
    attach_function :TFE_ContextStartStep, [:pointer], :void
    attach_function :TFE_ContextEndStep, [:pointer], :void

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/ops.h
    attach_function :TF_NewOpDefinitionBuilder, [:string], :pointer
    attach_function :TF_RegisterOpDefinition, [:pointer, :pointer], :void
    attach_function :TF_DeleteOpDefinitionBuilder, [:pointer], :void
    attach_function :TF_OpDefinitionBuilderAddAttr, [:pointer, :string], :void
    attach_function :TF_OpDefinitionBuilderAddInput, [:pointer, :string], :void
    attach_function :TF_OpDefinitionBuilderAddOutput, [:pointer, :string], :void
    attach_function :TF_OpDefinitionBuilderSetIsCommutative, [:pointer, :bool], :void
    attach_function :TF_OpDefinitionBuilderSetIsAggregate, [:pointer, :bool], :void
    attach_function :TF_OpDefinitionBuilderSetIsAggregate, [:pointer, :bool], :void
    attach_function :TF_OpDefinitionBuilderSetShapeInferenceFunction, [:pointer, :pointer], :void

    attach_function :TF_GraphToFunction, [:pointer, :string, :uchar,
                                          :int, :pointer,
                                          :int, :pointer,
                                          :int, :pointer,
                                          :pointer, :pointer, :string, :pointer], :pointer
    attach_function :TF_FunctionName, [:pointer], :strptr
    attach_function :TF_FunctionToFunctionDef, [:pointer, :pointer, :pointer], :strptr
    attach_function :TF_GraphCopyFunction, [:pointer, :pointer, :pointer, :pointer], :void

    attach_function :TF_GraphToGraphDef, [:pointer, :pointer, :pointer], :void

    attach_function :TF_NewImportGraphDefOptions, [], :pointer
    attach_function :TF_DeleteImportGraphDefOptions, [:pointer], :void
    attach_function :TF_ImportGraphDefOptionsSetPrefix, [:pointer, :string], :void
    attach_function :TF_ImportGraphDefOptionsSetDefaultDevice, [:pointer, :string], :void
    attach_function :TF_ImportGraphDefOptionsSetUniquifyNames, [:pointer, :uchar], :void
    attach_function :TF_ImportGraphDefOptionsSetUniquifyPrefix, [:pointer, :uchar], :void
    attach_function :TF_ImportGraphDefOptionsAddInputMapping, [:pointer, :string, :int, Output], :void
    attach_function :TF_ImportGraphDefOptionsRemapControlDependency, [:pointer, :string, :pointer], :void
    attach_function :TF_ImportGraphDefOptionsAddControlDependency, [:pointer, :pointer], :void
    attach_function :TF_ImportGraphDefOptionsAddReturnOutput, [:pointer,:string, :int], :void
    attach_function :TF_ImportGraphDefOptionsNumReturnOutputs, [:pointer], :int
    attach_function :TF_ImportGraphDefOptionsAddReturnOperation, [:pointer, :string], :void
    attach_function :TF_ImportGraphDefOptionsNumReturnOperations, [:pointer], :int

    attach_function :TF_GraphImportGraphDef, [:pointer, Buffer, :pointer, :pointer], :int
  end
end
