module Tensorflow
  module Strings
    class << self
      def as_string(input, precision: nil, scientific: nil, shortest: nil, width: nil, fill: nil)
        RawOps.as_string(input, precision: precision, scientific: scientific, shortest: shortest, width: width, fill: fill)
      end

      # def bytes_split
      # end

      # def format
      # end

      def join(inputs, separator: "")
        RawOps.string_join(inputs, separator: separator, n: inputs.size)
      end

      def length(input, unit: "BYTE")
        RawOps.string_length(input, unit: unit)
      end

      def lower(input)
        RawOps.string_lower(input.encode('UTF-8'), encoding: 'utf-8')
      end

      # def ngrams
      # end

      def reduce_join(inputs, reduction_indices, keep_dims: nil, separator: nil)
        RawOps.reduce_join(inputs, reduction_indices: reduction_indices, keep_dims: keep_dims, separator: separator)
      end

      def regex_full_match(input, pattern)
        RawOps.regex_full_match(input, pattern: pattern)
      end

      def regex_replace(input, pattern, rewrite, replace_global: nil)
        RawOps.regex_replace(input, pattern: pattern, rewrite: rewrite, replace_global: replace_global)
      end

      def split(split_dim, value, num_split: nil)
        RawOps.split(split_dim: split_dim, value: value, num_split: num_split)
      end

      def strip(input)
        RawOps.string_strip(input)
      end

      def substr(input, pos, len, unit: nil)
        RawOps.substr(input, pos, len, unit: unit)
      end

      # def to_hash_bucket
      # end

      # def to_hash_bucket_fast
      # end

      # def to_hash_bucket_strong
      # end

      def to_number(input, out_type: :float)
        RawOps.string_to_number(input, out_type: out_type)
      end

      def unicode_decode(input, input_encoding: nil, errors: nil, replacement_char: nil, replace_control_characters: nil)
        RawOps.unicode_decode(input, input_encoding: input_encoding, errors: errors, replacement_char: replacement_char, replace_control_characters: replace_control_characters)
      end

      def unicode_decode_with_offsets(input, input_encoding: nil, errors: nil, replacement_char: nil, replace_control_characters: nil)
        RawOps.unicode_decode_with_offsets(input, input_encoding: input_encoding, errors: errors, replacement_char: replacement_char, replace_control_characters: replace_control_characters)
      end

      def unicode_encode(input_values, input_splits, errors: nil, output_encoding: nil, replacement_char: nil)
        RawOps.unicode_encode(input_values: input_values, input_splits: input_splits, errors: errors, output_encoding: output_encoding, replacement_char: replacement_char)
      end

      def unicode_script(input)
        RawOps.unicode_script(input)
      end

      # def unicode_split
      # end

      # def unicode_split_with_offsets
      # end

      def unicode_transcode(input, input_encoding: nil, output_encoding: nil, errors: nil, replacement_char: nil, replace_control_characters: nil)
        RawOps.unicode_transcode(input, input_encoding: input_encoding, output_encoding: output_encoding, errors: errors, replacement_char: replacement_char, replace_control_characters: replace_control_characters)
      end

      # def unsorted_segment_join
      # end

      def upper(input, encoding: "")
        RawOps.string_upper(input, encoding: encoding)
      end
    end
  end
end
