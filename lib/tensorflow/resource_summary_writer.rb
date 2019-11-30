module Tensorflow
  class ResourceSummaryWriter
    attr_accessor :step
    attr_reader :initializer

    def initialize(shared_name: "", container: "")
      self.step = 1
      @resource = RawOps.summary_writer(shared_name: shared_name, container: container)
      @initializer = yield @resource
    end

    def create_summary_metadata(display_name, description)
      metadata = SummaryMetadata.new
      metadata.display_name = display_name
      metadata.summary_description = description
      metadata.plugin_data = SummaryMetadata.PluginData.new
      metadata.plugin_data.plugin_name = 'scalars'
    end

    def step=(value)
      @step = value.is_a?(Variable) ? value : Tensor.new(value, dtype: :int64)
    end

    def audio(tag, tensor, sample_rate)
      RawOps.write_audio_summary(@resource, self.step, tag, tensor, sample_rate)
    end

    def graph(graph)
      RawOps.write_graph_summary(@resource, self.step, graph.as_graph_def)
    end

    def histogram(tag, value, sample_rate)
      RawOps.write_histogram_summary(@resource, self.step, tag, value)
    end

    def image(tag, tensor, bad_color)
      RawOps.write_image_summary(@resource, self.step, tag, tensor, bad_color)
    end

    def proto(tag, tensor)
      RawOps.write_raw_proto_summary(@resource, self.step, tensor)
    end

    def scalar(tag, value, dtype: nil)
      RawOps.write_scalar_summary(@resource, self.step, tag, value, typeT: dtype)
    end

    def write(tag, value, metadata: "".b)
      value = Tensor.new(value)
      dtype ||= value.dtype

      RawOps.write_summary(@resource, step, value, tag, metadata, typeT: dtype)
    end

    def flush
      RawOps.flush_summary_writer(@resource)
    end

    def close
      RawOps.close_summary_writer(@resource)
    end
  end
end