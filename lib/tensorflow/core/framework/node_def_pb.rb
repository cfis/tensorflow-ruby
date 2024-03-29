# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/node_def.proto

require 'google/protobuf'

require 'tensorflow/core/framework/attr_value_pb'
Google::Protobuf::DescriptorPool.generated_pool.build do
  add_file("tensorflow/core/framework/node_def.proto", :syntax => :proto3) do
    add_message "tensorflow.NodeDef" do
      optional :name, :string, 1
      optional :op, :string, 2
      repeated :input, :string, 3
      optional :device, :string, 4
      map :attr, :string, :message, 5, "tensorflow.AttrValue"
      optional :experimental_debug_info, :message, 6, "tensorflow.NodeDef.ExperimentalDebugInfo"
    end
    add_message "tensorflow.NodeDef.ExperimentalDebugInfo" do
      repeated :original_node_names, :string, 1
      repeated :original_func_names, :string, 2
    end
  end
end

module Tensorflow
  NodeDef = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.NodeDef").msgclass
  NodeDef::ExperimentalDebugInfo = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.NodeDef.ExperimentalDebugInfo").msgclass
end
