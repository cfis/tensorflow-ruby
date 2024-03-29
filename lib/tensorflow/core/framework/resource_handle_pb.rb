# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/resource_handle.proto

require 'google/protobuf'

require 'tensorflow/core/framework/tensor_shape_pb'
require 'tensorflow/core/framework/types_pb'
Google::Protobuf::DescriptorPool.generated_pool.build do
  add_file("tensorflow/core/framework/resource_handle.proto", :syntax => :proto3) do
    add_message "tensorflow.ResourceHandleProto" do
      optional :device, :string, 1
      optional :container, :string, 2
      optional :name, :string, 3
      optional :hash_code, :uint64, 4
      optional :maybe_type_name, :string, 5
      repeated :dtypes_and_shapes, :message, 6, "tensorflow.ResourceHandleProto.DtypeAndShape"
    end
    add_message "tensorflow.ResourceHandleProto.DtypeAndShape" do
      optional :dtype, :enum, 1, "tensorflow.DataType"
      optional :shape, :message, 2, "tensorflow.TensorShapeProto"
    end
  end
end

module Tensorflow
  ResourceHandleProto = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.ResourceHandleProto").msgclass
  ResourceHandleProto::DtypeAndShape = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.ResourceHandleProto.DtypeAndShape").msgclass
end
