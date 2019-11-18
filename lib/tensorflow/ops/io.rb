module Tensorflow
  module IO
    def self.decode_and_crop_jpeg(contents, crop_window, channels: nil, ratio: nil, fancy_upscaling: nil, try_recover_truncated: nil, acceptable_fraction: nil, dct_method: nil)
      RawOps.decode_and_crop_jpeg(contents, crop_window, channels: channels, ratio: ratio, fancy_upscaling: fancy_upscaling, try_recover_truncated: try_recover_truncated, acceptable_fraction: acceptable_fraction, dct_method: dct_method)
    end

    def self.decode_base64(input)
      RawOps.decode_base64(input)
    end

    def self.decode_bmp(contents, channels: nil)
      RawOps.decode_bmp(contents, channels: channels)
    end

    def self.decode_compressed(bytes, compression_type: nil)
      RawOps.decode_compressed(bytes, compression_type: compression_type)
    end

    def self.decode_csv(records, record_defaults, field_delim: nil, use_quote_delim: nil, na_value: nil, select_cols: nil)
      RawOps.decode_csv(records, record_defaults, field_delim: field_delim, use_quote_delim: use_quote_delim, na_value: na_value, select_cols: select_cols)
    end

    def self.decode_gif(contents)
      RawOps.decode_gif(contents: contents)
    end

    # def self.decode_image
    # end

    def self.decode_jpeg(contents, channels: nil, ratio: nil, fancy_upscaling: nil, try_recover_truncated: nil, acceptable_fraction: nil, dct_method: nil)
      RawOps.decode_jpeg(contents: contents, channels: channels, ratio: ratio, fancy_upscaling: fancy_upscaling, try_recover_truncated: try_recover_truncated, acceptable_fraction: acceptable_fraction, dct_method: dct_method)
    end

    def self.decode_json_example(json_examples)
      RawOps.decode_json_example(json_examples: json_examples)
    end

    def self.decode_png(contents, channels: nil, dtype: nil)
      RawOps.decode_png(contents: contents, channels: channels, dtype: dtype)
    end

    # def self.decode_proto
    # end

    def self.decode_raw(bytes, out_type, little_endian: nil)
      RawOps.decode_raw(bytes, out_type: out_type, little_endian: little_endian)
    end

    def self.deserialize_many_sparse(serialized_sparse, dtype: nil)
      RawOps.deserialize_many_sparse(serialized_sparse: serialized_sparse, dtype: dtype)
    end

    def self.encode_base64(input, pad: nil)
      RawOps.encode_base64(input: input, pad: pad)
    end

    def self.encode_jpeg(image, format: nil, quality: nil, progressive: nil, optimize_size: nil, chroma_downsampling: nil, density_unit: nil, x_density: nil, y_density: nil, xmp_metadata: nil)
      RawOps.encode_jpeg(image: image, format: format, quality: quality, progressive: progressive, optimize_size: optimize_size, chroma_downsampling: chroma_downsampling, density_unit: density_unit, x_density: x_density, y_density: y_density, xmp_metadata: xmp_metadata)
    end

    def self.encode_proto(sizes, values, field_names: nil, message_type: nil, descriptor_source: nil)
      RawOps.encode_proto(sizes, values, field_names: field_names, message_type: message_type, descriptor_source: descriptor_source)
    end

    def self.extract_jpeg_shape(contents, output_type: nil)
      RawOps.extract_jpeg_shape(contents: contents, output_type: output_type)
    end

    def self.is_jpeg(contents)
      Image.is_jpeg(contents)
    end

    # def self.match_filenames_once
    # end

    def self.matching_files(pattern)
      RawOps.matching_files(pattern: pattern)
    end

    def self.parse_example(serialized, names, sparse_keys, dense_keys, dense_defaults, sparse_types: nil, dense_shapes: nil)
      RawOps.parse_example(serialized, names, sparse_keys, dense_keys, dense_defaults, sparse_types: sparse_types, dense_shapes: dense_shapes)
    end

    def self.parse_sequence_example(serialized, debug_name, context_dense_defaults, feature_list_dense_missing_assumed_empty: nil, context_sparse_keys: nil, context_dense_keys: nil, feature_list_sparse_keys: nil, feature_list_dense_keys: nil, context_sparse_types: nil, feature_list_dense_types: nil, context_dense_shapes: nil, feature_list_sparse_types: nil, feature_list_dense_shapes: nil)
      RawOps.parse_sequence_example(serialized, debug_name, context_dense_defaults, feature_list_dense_missing_assumed_empty: feature_list_dense_missing_assumed_empty, context_sparse_keys: context_sparse_keys, context_dense_keys: context_dense_keys, feature_list_sparse_keys: feature_list_sparse_keys, feature_list_dense_keys: feature_list_dense_keys, context_sparse_types: context_sparse_types, feature_list_dense_types: feature_list_dense_types, context_dense_shapes: context_dense_shapes, feature_list_sparse_types: feature_list_sparse_types, feature_list_dense_shapes: feature_list_dense_shapes)
    end

    def self.parse_single_example(serialized, dense_defaults, num_sparse: nil, sparse_keys: nil, dense_keys: nil, sparse_types: nil, dense_shapes: nil)
      RawOps.parse_single_example(serialized, dense_defaults, num_sparse: num_sparse, sparse_keys: sparse_keys, dense_keys: dense_keys, sparse_types: sparse_types, dense_shapes: dense_shapes)
    end

    def self.parse_single_sequence_example(serialized, feature_list_dense_missing_assumed_empty, context_sparse_keys, context_dense_keys, feature_list_sparse_keys, feature_list_dense_keys, context_dense_defaults, debug_name, context_sparse_types: nil, feature_list_dense_types: nil, context_dense_shapes: nil, feature_list_sparse_types: nil, feature_list_dense_shapes: nil)
      RawOps.parse_single_sequence_example(serialized, feature_list_dense_missing_assumed_empty, context_sparse_keys, context_dense_keys, feature_list_sparse_keys, feature_list_dense_keys, context_dense_defaults, debug_name)
    end

    def self.parse_tensor(serialized, out_type: nil)
      RawOps.parse_tensor(serialized: serialized, out_type: out_type)
    end

    def self.read_file(filename)
      RawOps.read_file(filename)
    end

    def self.serialize_many_sparse(sparse_indices, sparse_values, sparse_shape, out_type: nil)
      RawOps.serialize_many_sparse(sparse_indices, sparse_values, sparse_shape, out_type: out_type)
    end

    def self.serialize_sparse(sparse_indices, sparse_values, sparse_shape, out_type: nil)
      RawOps.serialize_sparse(sparse_indices, sparse_values, sparse_shape, out_type: out_type)
    end

    def self.serialize_tensor(tensor)
      RawOps.serialize_tensor(tensor)
    end

    def self.write_file(filename, contents)
      RawOps.write_file(filename, contents)
    end

    # def self.write_graph
    # end
  end
end
