module Tensorflow
  module Linalg
      # def self.adjoint
      # end

      # def self.band_part
      # end

      def self.cholesky(input)
        RawOps.cholesky(input: input)
      end

      # def self.cholesky_solve
      # end

      def self.cross(a, b)
        RawOps.cross(a, b)
      end

      # def self.det
      # end

      def self.diag(diagonal)
        RawOps.diag(diagonal: diagonal)
      end

      def self.diag_part(input)
        RawOps.diag_part(input: input)
      end

      # def self.eigh
      # end

      # def self.eigvalsh
      # end

      # def self.einsum
      # end

      # def self.expm
      # end

      def self.eye(num_rows, num_columns: nil)
        num_columns ||= num_rows
        zeros = Tensorflow.zeros([num_rows, num_columns])
        ones = Tensorflow.ones([num_rows])
        RawOps.matrix_set_diag(zeros, ones)
      end

      # def self.global_norm
      # end

      def self.inv(x)
        RawOps.inv(x: x)
      end

      # def self.l2_normalize
      # end

      # def self.logdet
      # end

      # def self.logm
      # end

      # def self.lstsq
      # end

      def self.lu(input, output_idx_type: nil)
        RawOps.lu(input: input, output_idx_type: output_idx_type)
      end

      def self.matmul(a, b, transpose_a: false, transpose_b: false)
        RawOps.mat_mul(a, b, transpose_a: transpose_a, transpose_b: transpose_b)
      end

      # def self.matrix_transpose
      # end

      # def self.matvec
      # end

      # def self.norm
      # end

      # def self.normalize
      # end

      def self.qr(input, full_matrices: nil)
        RawOps.qr(input: input, full_matrices: full_matrices)
      end

      # def self.set_diag
      # end

      # def self.slogdet
      # end

      # def self.solve
      # end

      # def self.sqrtm
      # end

      def self.svd(input, compute_uv: nil, full_matrices: nil)
        RawOps.svd(input: input, compute_uv: compute_uv, full_matrices: full_matrices)
      end

      # def self.tensor_diag
      # end

      # def self.tensor_diag_part
      # end

      # def self.tensordot
      # end

      # def self.trace
      # end

      # def self.triangular_solve
      # end

      # def self.tridiagonal_matmul
      # end

      def self.tridiagonal_solve(diagonals, rhs, partial_pivoting: nil)
        RawOps.tridiagonal_solve(diagonals, rhs, partial_pivoting: partial_pivoting)
      end
    end
end
