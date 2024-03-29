# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/verifier_config.proto

require 'google/protobuf'

Google::Protobuf::DescriptorPool.generated_pool.build do
  add_file("tensorflow/core/protobuf/verifier_config.proto", :syntax => :proto3) do
    add_message "tensorflow.VerifierConfig" do
      optional :verification_timeout_in_ms, :int64, 1
      optional :structure_verifier, :enum, 2, "tensorflow.VerifierConfig.Toggle"
    end
    add_enum "tensorflow.VerifierConfig.Toggle" do
      value :DEFAULT, 0
      value :ON, 1
      value :OFF, 2
    end
  end
end

module Tensorflow
  VerifierConfig = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.VerifierConfig").msgclass
  VerifierConfig::Toggle = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("tensorflow.VerifierConfig.Toggle").enummodule
end
