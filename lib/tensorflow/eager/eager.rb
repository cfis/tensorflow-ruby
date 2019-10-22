module Tensorflow
  module Eager
    def self.convert_to_tensor_handle(value, dtype: nil)
      case value
        when TensorHandle
          value
        when Data::Dataset
          value.variant_tensor
        when Tensor
          TensorHandle.new(value)
        when Variable
          value.value_handle
        else
          tensor = Tensor.new(value, dtype: dtype)
          if dtype && tensor.dtype != dtype
            raise StandardError, "Tensor conversion requested dtype #{dtype} for Tensor with dtype #{value.dtype}"
          end
          TensorHandle.new(tensor)
      end
    end

  end
end