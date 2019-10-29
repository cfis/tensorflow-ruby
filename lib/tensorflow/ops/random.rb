module Tensorflow
  module Random
    class << self
      def normal(shape, mean: 0.0, stddev: 1.0, dtype: :float, seed: nil, name: nil)
        shape_tensor = Tensor.new(shape)
        mean_tensor = Tensor.new(mean, dtype: dtype)
        stddev_tensor = Tensor.new(stddev, dtype: dtype)

        random = RawOps.random_standard_normal(shape, seed: nil, seed2: nil, dtype: dtype)
        mul = random * stddev_tensor
        Math.add(mul, mean_tensor)
      end
    end
  end
end