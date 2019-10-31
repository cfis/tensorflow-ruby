# keep in alphabetical order
module Tensorflow
  module Ops
    def cast(x, dtype)
      RawOps.cast(x, dstt: dtype)
    end

    def const(value: nil, dtype: nil)
      dtype ||= Utils.infer_dtype(value)
      RawOps.const(value: value, dtype: dtype)
    end

    def expand_dims(input, axis)
      RawOps.expand_dims(input, axis)
    end

    def fill(dims, value)
      RawOps.fill(dims, value)
    end

    def identity(input)
      RawOps.identity(input)
    end

    def ones(dims)
      fill(dims, 1)
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

    def split(split_dim, value, num_split: nil, typeT: nil)
      RawOps.split(split_dim, value, num_split: num_split, typeT: typeT)
    end

    def split_v(value, size_splits, split_dim=0, num_split: nil, typeT: nil, tlen: nil)
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

    def zeros(dims)
      fill(dims, 0)
    end

    def zeros_like(x)
      RawOps.zeros_like(x)
    end
  end
end
