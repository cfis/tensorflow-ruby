module Tensorflow
  module Math
    class << self
      def abs(x)
        RawOps.abs(x)
      end

      # def accumulate_n
      # end

      def acos(x)
        RawOps.acos(x)
      end

      def acosh(x)
        RawOps.acosh(x)
      end

      def add(x, y)
        RawOps.add(x, y)
      end

      def add_n(inputs)
        RawOps.add_n(inputs, n: inputs.length)
      end

      def angle(input)
        RawOps.angle(input)
      end

      # def argmax
      # end

      # def argmin
      # end

      def asin(x)
        RawOps.asin(x)
      end

      def asinh(x)
        RawOps.asinh(x)
      end

      def atan(x)
        RawOps.atan(x)
      end

      def atan2(y, x)
        RawOps.atan2(y, x)
      end

      def atanh(x)
        RawOps.atanh(x)
      end

      # def bessel_i0
      # end

      def bessel_i0e(x)
        RawOps.bessel_i0e(x)
      end

      # def bessel_i1
      # end

      def bessel_i1e(x)
        RawOps.bessel_i1e(x)
      end

      def betainc(a, b, x)
        RawOps.betainc(a, b, x)
      end

      def bincount(arr, size, weights)
        RawOps.bincount(arr, size, weights)
      end

      def ceil(x)
        RawOps.ceil(x)
      end

      # def confusion_matrix
      # end

      def conj(input)
        RawOps.conj(input)
      end

      def cos(x)
        RawOps.cos(x)
      end

      def cosh(x)
        RawOps.cosh(x)
      end

      # def count_nonzero
      # end

      def cumprod(x, axis, exclusive: nil, reverse: nil)
        RawOps.cumprod(x, axis, exclusive: exclusive, reverse: reverse)
      end

      def cumsum(x, axis, exclusive: nil, reverse: nil)
        RawOps.cumsum(x, axis, exclusive: exclusive, reverse: reverse)
      end

      # def cumulative_logsumexp
      # end

      def digamma(x)
        RawOps.digamma(x)
      end

      def divide(x, y)
        RawOps.div(x, y)
      end

      # def divide_no_nan
      # end

      def equal(x, y)
        RawOps.equal(x, y)
      end

      def erf(x)
        RawOps.erf(x)
      end

      def erfc(x)
        RawOps.erfc(x)
      end

      def exp(x)
        RawOps.exp(x)
      end

      def expm1(x)
        RawOps.expm1(x)
      end

      def floor(x)
        RawOps.floor(x)
      end

      def floordiv(x, y)
        RawOps.floor_div(x, y)
      end

      def floormod(x, y)
        RawOps.floor_mod(x, y)
      end

      def greater(x, y)
        RawOps.greater(x, y)
      end

      def greater_equal(x, y)
        RawOps.greater_equal(x, y)
      end

      def igamma(a, x)
        RawOps.igamma(a, x)
      end

      def igammac(a, x)
        RawOps.igammac(a, x)
      end

      def imag(input)
        RawOps.imag(input)
      end

      def in_top_k(predictions, targets, k: nil)
        RawOps.in_top_k(predictions, targets, k: k)
      end

      def invert_permutation(x)
        RawOps.invert_permutation(x)
      end

      def is_finite(x)
        RawOps.is_finite(x)
      end

      def is_inf(x)
        RawOps.is_inf(x)
      end

      def is_nan(x)
        RawOps.is_nan(x)
      end

      # def is_non_decreasing
      # end

      # def is_strictly_increasing
      # end

      # def l2_normalize
      # end

      # def lbeta
      # end

      def less(x, y)
        RawOps.less(x, y)
      end

      def less_equal(x, y)
        RawOps.less_equal(x, y)
      end

      def lgamma(x)
        RawOps.lgamma(x)
      end

      def log(x)
        RawOps.log(x)
      end

      def log1p(x)
        RawOps.log1p(x)
      end

      def log_sigmoid(x)
        negative(RawOps.softplus(-x))
      end

      def log_softmax(logits)
        RawOps.log_softmax(logits: logits)
      end

      def logical_and(x, y)
        RawOps.logical_and(x, y)
      end

      def logical_not(x)
        RawOps.logical_not(x)
      end

      def logical_or(x, y)
        RawOps.logical_or(x, y)
      end

      def logical_xor(x, y)
        logical_and(logical_or(x, y), logical_not(logical_and(x, y)))
      end

      def maximum(x, y)
        RawOps.maximum(x, y)
      end

      def minimum(x, y)
        RawOps.minimum(x, y)
      end

      def mod(x, y)
        RawOps.mod(x, y)
      end

      def multiply(x, y)
        RawOps.mul(x, y)
      end

      def multiply_no_nan(x, y)
        RawOps.mul_no_nan(x, y)
      end

      def negative(x)
        RawOps.neg(x)
      end

      # def nextafter
      # end

      def not_equal(x, y)
        RawOps.not_equal(x, y)
      end

      def polygamma(a, x)
        RawOps.polygamma(a, x)
      end

      # def polyval
      # end

      def pow(x, y)
        RawOps.pow(x, y)
      end

      def real(input)
        RawOps.real(input)
      end

      def reciprocal(x)
        RawOps.reciprocal(x)
      end

      # def reciprocal_no_nan
      # end

      # def reduce_all
      # end

      def reduce_any(input, axis: nil, keepdims: false)
        axis ||= reduction_dims(input)
        RawOps.any(input, axis, keep_dims: keepdims)
      end

      # def reduce_euclidean_norm
      # end

      # def reduce_logsumexp
      # end

      def reduce_max(input, axis: nil, keepdims: false)
        axis ||= reduction_dims(input)
        RawOps.max(input, axis, keep_dims: keepdims)
      end

      def reduce_mean(input, axis: nil, keepdims: false)
        axis ||= reduction_dims(input)
        RawOps.mean(input, axis, keep_dims: keepdims)
      end

      def reduce_min(input, axis: nil, keepdims: false)
        axis ||= reduction_dims(input)
        RawOps.min(input, axis, keep_dims: keepdims)
      end

      def reduce_prod(input, axis: nil, keepdims: false)
        axis ||= reduction_dims(input)
        RawOps.prod(input, axis, keep_dims: keepdims)
      end

      def reduce_std(input, axis: nil, keepdims: false)
        variance = reduce_variance(input, axis: axis, keepdims: keepdims)
        sqrt(variance)
      end

      def reduce_sum(input, axis: nil, keepdims: false)
        axis ||= reduction_dims(input)
        RawOps.sum(input, axis, keep_dims: keepdims)
      end

      def reduce_variance(input, axis: nil, keepdims: false)
        means = reduce_mean(input, axis: axis, keepdims: true)
        squared_deviations = RawOps.square(input - means)
        reduce_mean(squared_deviations, axis: axis, keepdims: keepdims)
      end

      def rint(x)
        RawOps.rint(x)
      end

      def round(x)
        RawOps.round(x)
      end

      def rsqrt(x)
        RawOps.rsqrt(x)
      end

      # def scalar_mul
      # end

      def segment_max(data, segment_ids)
        RawOps.segment_max(data, segment_ids)
      end

      def segment_mean(data, segment_ids)
        RawOps.segment_mean(data, segment_ids)
      end

      def segment_min(data, segment_ids)
        RawOps.segment_min(data, segment_ids)
      end

      def segment_prod(data, segment_ids)
        RawOps.segment_prod(data, segment_ids)
      end

      def segment_sum(data, segment_ids)
        RawOps.segment_sum(data, segment_ids)
      end

      def sigmoid(x)
        RawOps.sigmoid(x)
      end

      def sign(x)
        RawOps.sign(x)
      end

      def sin(x)
        RawOps.sin(x)
      end

      def sinh(x)
        RawOps.sinh(x)
      end

      def softmax(logits)
        RawOps.softmax(logits: logits)
      end

      def softplus(features)
        RawOps.softplus(features: features)
      end

      def softsign(features)
        RawOps.softsign(features: features)
      end

      def sqrt(x)
        RawOps.sqrt(x)
      end

      def square(x)
        RawOps.square(x)
      end

      def squared_difference(x, y)
        RawOps.squared_difference(x, y)
      end

      def subtract(x, y)
        RawOps.sub(x, y)
      end

      def tan(x)
        RawOps.tan(x)
      end

      def tanh(x)
        RawOps.tanh(x)
      end

      def top_k(input, k: nil, sorted: nil)
        RawOps.top_k(input, k: k, sorted: sorted)
      end

      # def truediv
      # end

      def unsorted_segment_max(data, segment_ids, num_segments)
        RawOps.unsorted_segment_max(data, segment_ids, num_segments: num_segments)
      end

      # def unsorted_segment_mean
      # end

      def unsorted_segment_min(data, segment_ids, num_segments)
        RawOps.unsorted_segment_min(data, segment_ids, num_segments: num_segments)
      end

      def unsorted_segment_prod(data, segment_ids, num_segments)
        RawOps.unsorted_segment_prod(data, segment_ids, num_segments: num_segments)
      end

      # def unsorted_segment_sqrt_n
      # end

      def unsorted_segment_sum(data, segment_ids, num_segments)
        RawOps.unsorted_segment_sum(data, segment_ids, num_segments: num_segments)
      end

      def xdivy(x, y)
        RawOps.xdivy(x, y)
      end

      def xlogy(x, y)
        RawOps.xlogy(x, y)
      end

      # def zero_fraction
      # end

      def zeta(x, q)
        RawOps.zeta(x, q)
      end

      private

      def reduction_dims(input)
        rank = Tensorflow.rank(input)
        range = Tensorflow.range(0, rank)
      end
    end
  end
end
