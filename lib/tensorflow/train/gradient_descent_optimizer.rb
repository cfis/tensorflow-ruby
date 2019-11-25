# Based on code from https://github.com/jedld/tensor_stream

module Tensorflow
  module Train
    class GradientDescentOptimizer < Optimizer
      attr_accessor :learning_rate

      def initialize(learning_rate, use_locking: false, name: "GradientDescent")
        @learning_rate = learning_rate
        @learning_rate_tensor = nil
        super(name: name, use_locking: use_locking)
      end

      protected

      def prepare
        learning_rate = call_if_callable(@learning_rate)
        @learning_rate_tensor = Tensorflow.constant(learning_rate, name: "learning_rate")
      end

      def apply_dense(grad, var)
        dtype = grad.output_types.first
        learning_rate = if @learning_rate_tensor.output_types.first == dtype
                          @learning_rate_tensor
                        else
                          Tensorflow.cast(@learning_rate_tensor, dtype)
                        end

        RawOps.resource_apply_gradient_descent(var, learning_rate, grad)
      end
    end
  end
end
