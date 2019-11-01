module Tensorflow
  module Random
    def self.normal(shape, mean: 0.0, stddev: 1.0, dtype: :float, seed: nil, name: nil)
      random = RawOps.random_standard_normal(shape, seed: nil, seed2: nil, dtype: dtype)
      mul = (random * stddev) + mean
    end

    def self.uniform(shape, seed: 0, seed2: 0, dtype: :float, typeT: nil)
      RawOps.random_uniform(shape, seed: seed, seed2: seed2, dtype: dtype, typeT: typeT)
    end
  end
end