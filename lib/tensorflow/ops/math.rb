module Tensorflow
  module Math
    class << self
      def abs(x, dtype: nil)
        RawOps.abs(x, typeT: dtype)
      end

      # def accumulate_n
      # end

      def acos(x, dtype: nil)
        RawOps.acos(x, typeT: dtype)
      end

      def acosh(x, dtype: nil)
        RawOps.acosh(x, typeT: dtype)
      end

      def add(x, y, dtype: nil)
        RawOps.add(x, y, typeT: dtype)
      end

      def add_n(inputs, dtype: nil)
        RawOps.add_n(inputs, n: inputs.length, typeT: dtype)
      end

      def angle(input, dtype: nil)
        RawOps.angle(input, typeT: dtype)
      end

      # def argmax
      # end

      # def argmin
      # end

      def asin(x, dtype: nil)
        RawOps.asin(x, typeT: dtype)
      end

      def asinh(x, dtype: nil)
        RawOps.asinh(x, typeT: dtype)
      end

      def atan(x, dtype: nil)
        RawOps.atan(x, typeT: dtype)
      end

      def atan2(y, x, dtype: nil)
        RawOps.atan2(y, x, typeT: dtype)
      end

      def atanh(x, dtype: nil)
        RawOps.atanh(x, typeT: dtype)
      end

      # def bessel_i0
      # end

      def bessel_i0e(x, dtype: nil)
        RawOps.bessel_i0e(x, typeT: dtype)
      end

      # def bessel_i1
      # end

      def bessel_i1e(x, dtype: nil)
        RawOps.bessel_i1e(x, typeT: dtype)
      end

      def betainc(a, b, x, dtype: nil)
        RawOps.betainc(a, b, x, typeT: dtype)
      end

      def bincount(arr, size, weights, dtype: nil)
        RawOps.bincount(arr, size, weights, typeT: dtype)
      end

      def ceil(x, dtype: nil)
        RawOps.ceil(x, typeT: dtype)
      end

      # def confusion_matrix
      # end

      def conj(input, dtype: nil)
        RawOps.conj(input, typeT: dtype)
      end

      def cos(x, dtype: nil)
        RawOps.cos(x, typeT: dtype)
      end

      def cosh(x, dtype: nil)
        RawOps.cosh(x, typeT: dtype)
      end

      # def count_nonzero
      # end

      def cumprod(x, axis, exclusive: nil, reverse: nil, dtype: nil)
        RawOps.cumprod(x, axis, exclusive: exclusive, reverse: reverse, typeT: dtype)
      end

      def cumsum(x, axis, exclusive: nil, reverse: nil, dtype: nil)
        RawOps.cumsum(x, axis, exclusive: exclusive, reverse: reverse, typeT: dtype)
      end

      # def cumulative_logsumexp
      # end

      def digamma(x, dtype: nil)
        RawOps.digamma(x, typeT: dtype)
      end

      def divide(x, y, dtype: nil)
        RawOps.div(x, y, typeT: dtype)
      end

      # def divide_no_nan
      # end

      def equal(x, y, dtype: nil)
        RawOps.equal(x, y, typeT: dtype)
      end

      def erf(x, dtype: nil)
        RawOps.erf(x, typeT: dtype)
      end

      def erfc(x, dtype: nil)
        RawOps.erfc(x, typeT: dtype)
      end

      def exp(x, dtype: nil)
        RawOps.exp(x, typeT: dtype)
      end

      def expm1(x, dtype: nil)
        RawOps.expm1(x, typeT: dtype)
      end

      def floor(x, dtype: nil)
        RawOps.floor(x, typeT: dtype)
      end

      def floordiv(x, y, dtype: nil)
        RawOps.floor_div(x, y, typeT: dtype)
      end

      def floormod(x, y, dtype: nil)
        RawOps.floor_mod(x, y, typeT: dtype)
      end

      def greater(x, y, dtype: nil)
        RawOps.greater(x, y, typeT: dtype)
      end

      def greater_equal(x, y, dtype: nil)
        RawOps.greater_equal(x, y, typeT: dtype)
      end

      def igamma(a, x, dtype: nil)
        RawOps.igamma(a, x, typeT: dtype)
      end

      def igammac(a, x, dtype: nil)
        RawOps.igammac(a, x, typeT: dtype)
      end

      def imag(input, dtype: nil)
        RawOps.imag(input, typeT: dtype)
      end

      def in_top_k(predictions, targets, k=nil, dtype: nil)
        RawOps.in_top_kv2(predictions, targets, k, typeT: dtype)
      end

      def invert_permutation(x, dtype: nil)
        RawOps.invert_permutation(x, typeT: dtype)
      end

      def is_finite(x, dtype: nil)
        RawOps.is_finite(x, typeT: dtype)
      end

      def is_inf(x, dtype: nil)
        RawOps.is_inf(x, typeT: dtype)
      end

      def is_nan(x, dtype: nil)
        RawOps.is_nan(x, typeT: dtype)
      end

      # def is_non_decreasing
      # end

      # def is_strictly_increasing
      # end

      # def l2_normalize
      # end

      # def lbeta
      # end

      def less(x, y, dtype: nil)
        RawOps.less(x, y, typeT: dtype)
      end

      def less_equal(x, y, dtype: nil)
        RawOps.less_equal(x, y, typeT: dtype)
      end

      def lgamma(x, dtype: nil)
        RawOps.lgamma(x, typeT: dtype)
      end

      def log(x, dtype: nil)
        RawOps.log(x, typeT: dtype)
      end

      def log1p(x, dtype: nil)
        RawOps.log1p(x, typeT: dtype)
      end

      def log_sigmoid(x, dtype: nil)
        negative(RawOps.softplus(-x, typeT: nil), dtype: dtype)
      end

      def log_softmax(logits, dtype: nil)
        RawOps.log_softmax(logits: logits, typeT: dtype)
      end

      def logical_and(x, y, dtype: nil)
        RawOps.logical_and(x, y)
      end

      def logical_not(x, dtype: nil)
        RawOps.logical_not(x)
      end

      def logical_or(x, y, dtype: nil)
        RawOps.logical_or(x, y)
      end

      def logical_xor(x, y, dtype: nil)
        logical_and(logical_or(x, y, dtype: nil), logical_not(logical_and(x, y, dtype: nil), dtype: nil))
      end

      def maximum(x, y, dtype: nil)
        RawOps.maximum(x, y, typeT: dtype)
      end

      def minimum(x, y, dtype: nil)
        RawOps.minimum(x, y, typeT: dtype)
      end

      def mod(x, y, dtype: nil)
        RawOps.mod(x, y, typeT: dtype)
      end

      def multiply(x, y, dtype: nil)
        RawOps.mul(x, y, typeT: dtype)
      end

      def multiply_no_nan(x, y, dtype: nil)
        RawOps.mul_no_nan(x, y, typeT: dtype)
      end

      def negative(x, dtype: nil)
        RawOps.neg(x, typeT: dtype)
      end

      # def nextafter
      # end

      def not_equal(x, y, dtype: nil)
        RawOps.not_equal(x, y, typeT: dtype)
      end

      def polygamma(a, x, dtype: nil)
        RawOps.polygamma(a, x, typeT: dtype)
      end

      # def polyval
      # end

      def pow(x, y, dtype: nil)
        RawOps.pow(x, y, typeT: dtype)
      end

      def real(input, dtype: nil)
        RawOps.real(input, typeT: dtype)
      end

      def reciprocal(x, dtype: nil)
        RawOps.reciprocal(x, typeT: dtype)
      end

      # def reciprocal_no_nan
      # end

      # def reduce_all
      # end

      def reduce_any(input, axis: nil, keepdims: false, dtype: nil)
        axis ||= reduction_dims(input, dtype: dtype)
        RawOps.any(input, axis, keep_dims: keepdims)
      end

      # def reduce_euclidean_norm
      # end

      # def reduce_logsumexp
      # end

      def reduce_max(input, axis: nil, keepdims: false, dtype: nil)
        axis ||= reduction_dims(input, dtype: dtype)
        RawOps.max(input, axis, keep_dims: keepdims, typeT: dtype)
      end

      def reduce_mean(input, axis: nil, keepdims: false, dtype: nil)
        axis ||= reduction_dims(input, dtype: dtype)
        RawOps.mean(input, axis, keep_dims: keepdims, typeT: dtype)
      end

      def reduce_min(input, axis: nil, keepdims: false, dtype: nil)
        axis ||= reduction_dims(input, dtype: dtype)
        RawOps.min(input, axis, keep_dims: keepdims, typeT: dtype)
      end

      def reduce_prod(input, axis: nil, keepdims: false, dtype: nil)
        axis ||= reduction_dims(input, dtype: dtype)
        RawOps.prod(input, axis, keep_dims: keepdims, typeT: dtype)
      end

      def reduce_std(input, axis: nil, keepdims: false, dtype: nil)
        variance = reduce_variance(input, axis: axis, keepdims: keepdims, dtype: dtype)
        sqrt(variance, dtype: dtype)
      end

      def reduce_sum(input, axis: nil, keepdims: false, dtype: nil)
        axis ||= reduction_dims(input, dtype: dtype)
        RawOps.sum(input, axis, keep_dims: keepdims, typeT: dtype)
      end

      def reduce_variance(input, axis: nil, keepdims: false, dtype: nil)
        means = reduce_mean(input, axis: axis, keepdims: true, dtype: dtype)
        squared_deviations = RawOps.square(input - means, typeT: dtype)
        reduce_mean(squared_deviations, axis: axis, keepdims: keepdims, dtype: dtype)
      end

      def rint(x, dtype: nil)
        RawOps.rint(x, typeT: dtype)
      end

      def round(x, dtype: nil)
        RawOps.round(x, typeT: dtype)
      end

      def rsqrt(x, dtype: nil)
        RawOps.rsqrt(x, typeT: dtype)
      end

      # def scalar_mul
      # end

      def segment_max(data, segment_ids, dtype: nil)
        RawOps.segment_max(data, segment_ids, typeT: dtype)
      end

      def segment_mean(data, segment_ids, dtype: nil)
        RawOps.segment_mean(data, segment_ids, typeT: dtype)
      end

      def segment_min(data, segment_ids, dtype: nil)
        RawOps.segment_min(data, segment_ids, typeT: dtype)
      end

      def segment_prod(data, segment_ids, dtype: nil)
        RawOps.segment_prod(data, segment_ids, typeT: dtype)
      end

      def segment_sum(data, segment_ids, dtype: nil)
        RawOps.segment_sum(data, segment_ids, typeT: dtype)
      end

      def sigmoid(x, dtype: nil)
        RawOps.sigmoid(x, typeT: dtype)
      end

      def sign(x, dtype: nil)
        RawOps.sign(x, typeT: dtype)
      end

      def sin(x, dtype: nil)
        RawOps.sin(x, typeT: dtype)
      end

      def sinh(x, dtype: nil)
        RawOps.sinh(x, typeT: dtype)
      end

      def softmax(logits, dtype: nil)
        RawOps.softmax(logits: logits, typeT: dtype)
      end

      def softplus(features, dtype: nil)
        RawOps.softplus(features: features, typeT: dtype)
      end

      def softsign(features, dtype: nil)
        RawOps.softsign(features: features, typeT: dtype)
      end

      def sqrt(x, dtype: nil)
        RawOps.sqrt(x, typeT: dtype)
      end

      def square(x, dtype: nil)
        RawOps.square(x, typeT: dtype)
      end

      def squared_difference(x, y, dtype: nil)
        RawOps.squared_difference(x, y, typeT: dtype)
      end

      def subtract(x, y, dtype: nil)
        RawOps.sub(x, y, typeT: dtype)
      end

      def tan(x, dtype: nil)
        RawOps.tan(x, typeT: dtype)
      end

      def tanh(x, dtype: nil)
        RawOps.tanh(x, typeT: dtype)
      end

      def top_k(input, k: nil, sorted: nil, dtype: nil)
        RawOps.top_k(input, k: k, sorted: sorted, typeT: dtype)
      end

      # def truediv
      # end

      def unsorted_segment_max(data, segment_ids, num_segments, dtype: nil)
        RawOps.unsorted_segment_max(data, segment_ids, num_segments: num_segments, typeT: dtype)
      end

      # def unsorted_segment_mean
      # end

      def unsorted_segment_min(data, segment_ids, num_segments, dtype: nil)
        RawOps.unsorted_segment_min(data, segment_ids, num_segments: num_segments, typeT: dtype)
      end

      def unsorted_segment_prod(data, segment_ids, num_segments, dtype: nil)
        RawOps.unsorted_segment_prod(data, segment_ids, num_segments: num_segments, typeT: dtype)
      end

      # def unsorted_segment_sqrt_n
      # end

      def unsorted_segment_sum(data, segment_ids, num_segments, dtype: nil)
        RawOps.unsorted_segment_sum(data, segment_ids, num_segments: num_segments, typeT: dtype)
      end

      def xdivy(x, y, dtype: nil)
        RawOps.xdivy(x, y, typeT: dtype)
      end

      def xlogy(x, y, dtype: nil)
        RawOps.xlogy(x, y, typeT: dtype)
      end

      # def zero_fraction
      # end

      def zeta(x, q, dtype: nil)
        RawOps.zeta(x, q, typeT: dtype)
      end

      private

      def reduction_dims(input, dtype: nil)
        rank = Tensorflow.rank(input, typeT: dtype)
        range = Tensorflow.range(0, rank)
      end
    end
  end
end
