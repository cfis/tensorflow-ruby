module Tensorflow
  module Ops
    def self.broadcast_mul(vector, matrix)
      vector = Tensorflow.expand_dims(vector, -1)
      vector * matrix
    end

    Graph::Gradients.register('SparseSoftmaxCrossEntropyWithLogits') do |gradient, outputs, inputs|
      message = <<~EOS
        Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused
        implementation's interaction with tf.gradients()
      EOS

      graph = gradient.graph
      operation = outputs[0].operation
      sparse_softmax_grad_without_gradient = Tensorflow.prevent_gradient(operation[1], message: message)
      op = Ops.broadcast_mul(gradient, sparse_softmax_grad_without_gradient)
      op.outputs
    end
  end
end
