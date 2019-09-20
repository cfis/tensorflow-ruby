module TensorFlow
  module Ops
    def matmul(x, y)
      execute("MatMul", [x, y])
    end

    def transpose(x, perm: [1, 0])
      execute("Transpose", [x, perm])
    end

    def zeros(dims)
      fill(dims, 0)
    end

    def ones(dims)
      fill(dims, 1)
    end

    def eye(num_rows,  num_columns: nil)
      num_columns ||= num_rows
      zeros = self.zeros([num_rows, num_columns])
      ones = self.ones([num_rows])
      matrix_set_diag(zeros, ones)
    end
  end
end