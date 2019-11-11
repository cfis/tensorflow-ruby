# keep in alphabetical order
module Tensorflow
  module Ops
    def cast(x, source_dtype: nil, destination_dtype: nil, truncate: false)
      RawOps.cast(x, srct: source_dtype, dstt: destination_dtype, truncate: truncate)
    end

    def constant(value, dtype: nil, shape: [], name: 'Const')
      tensor = value.is_a?(Tensor) ? value : Tensor.new(value, dtype: dtype, shape: shape)
      RawOps.const(value: tensor, dtype: tensor.dtype, name: name)
    end

    def expand_dims(input, axis)
      RawOps.expand_dims(input, axis)
    end

    def fill(dims, value, dtype: nil)
      RawOps.fill(dims, value, typeT: dtype)
    end

    def identity(input)
      RawOps.identity(input)
    end

    def ones(dims)
      fill(dims, 1)
    end

    def pack(values, n: nil, typeT: nil, axis: 0)
      typeT ||= TensorData.figure_dtype(values)
      n ||= values.count
      RawOps.pack(values, n: n, typeT: typeT, axis: axis)
    end

    def placeholder(dtype, name: 'Placeholder', shape: nil)
      RawOps.placeholder(dtype: dtype, shape: shape, name: name)
    end

    def rank(input, typeT: nil)
      RawOps.rank(input, typeT: typeT)
    end

    def range(start, limit = nil, delta = 1)
      unless limit
        limit = start
        start = 0
      end
      RawOps.range(start, limit, delta)
    end

    def reshape(tensor, shape)
      RawOps.reshape(tensor, shape)
    end

    def shape(input, out_type)
      RawOps.shape(input, out_type: out_type)
    end

    def split(value, split_dim, num_split: nil, typeT: nil)
      RawOps.split(split_dim, value, num_split: num_split, typeT: typeT)
    end

    def split_v(value, size_splits, split_dim=0, num_split: nil, typeT: nil, tlen: nil)
      num_split ||= size_splits.length
      RawOps.split_v(value, size_splits, split_dim, num_split: num_split, typeT: typeT, tlen: tlen)
    end

    def squeeze(input, axis: nil)
      RawOps.squeeze(input, squeeze_dims: axis)
    end

    def timestamp
      RawOps.timestamp
    end

    def transpose(x, perm: [1, 0])
      RawOps.transpose(x, perm)
    end

    def zeros(dims, dtype: nil)
      const = self.constant(0, dtype: dtype)
      fill(dims, const, dtype: dtype)
    end

    def zeros_like(x)
      RawOps.zeros_like(x)
    end
  end
end
