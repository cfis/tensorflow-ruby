# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/tensor.proto

require 'google/protobuf'

require 'tensorflow/core/framework/resource_handle_pb'
require 'tensorflow/core/framework/tensor_shape_pb'
require 'tensorflow/core/framework/types_pb'
Google::Protobuf::DescriptorPool.generated_pool.build do
  add_file("tensorflow/core/framework/tensor.proto", :syntax => :proto3) do
    add_message "tensorflow.TensorProto" do
      optional :dtype, :enum, 1, "tensorflow.DataType"
      optional :tensor_shape, :message, 2, "tensorflow.TensorShapeProto"
      optional :version_number, :int32, 3
      optional :tensor_content, :bytes, 4
      repeated :half_val, :int32, 13
      repeated :float_val, :float, 5
      repeated :double_val, :double, 6
      repeated :int_val, :int32, 7
      repeated :string_val, :bytes, 8
      repeated :scomplex_val, :float, 9
      repeated :int64_val, :int64, 10
      repeated :bool_val, :bool, 11
      repeated :dcomplex_val, :double, 12
      repeated :resource_handle_val, :message, 14, "tensorflow.ResourceHandleProto"
      repeated :variant_val, :message, 15, "tensorflow.VariantTensorDataProto"
      repeated :uint32_val, :uint32, 16
      repeated :uint64_val, :uint64, 17
    end
    add_message "tensorflow.VariantTensorDataProto" do
      optional :type_name, :string, 1
      optional :metadata, :bytes, 2
      repeated :tensors, :message, 3, "tensorflow.TensorProto"
    end
  end
end

module Tensorflow
  TensorProto = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.TensorProto").msgclass
  VariantTensorDataProto = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.VariantTensorDataProto").msgclass
end
