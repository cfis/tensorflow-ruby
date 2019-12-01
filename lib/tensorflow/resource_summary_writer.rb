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

    def audio(tag, tensor, sample_rate, max_outputs: 3)
      tensor = Tensor.from_value(tensor, dtype: :float)
      result = RawOps.write_audio_summary(@resource, self.step, tag, tensor, sample_rate, max_outputs: max_outputs)
      ExecutionContext.current.add_to_collection(Graph::GraphKeys::SUMMARY_COLLECTION, result)
      result
    end

    def graph(graph)
      RawOps.write_graph_summary(@resource, self.step, graph.as_graph_def)
    end

    def histogram(tag, values)
      result = RawOps.write_histogram_summary(@resource, self.step, tag, values)
      ExecutionContext.current.add_to_collection(Graph::GraphKeys::SUMMARY_COLLECTION, result)
      result
    end

    def image(tag, tensor, bad_color=nil)
      bad_color ||= Tensor.new([255, 0, 0, 255], dtype: :uint8)
      result = RawOps.write_image_summary(@resource, self.step, tag, tensor, bad_color)
      ExecutionContext.current.add_to_collection(Graph::GraphKeys::SUMMARY_COLLECTION, result)
      result
    end

    def proto(tag, tensor)
      result = RawOps.write_raw_proto_summary(@resource, self.step, tensor)
      ExecutionContext.current.add_to_collection(Graph::GraphKeys::SUMMARY_COLLECTION, result)
      result
    end

    def scalar(tag, value, dtype: nil)
      result = RawOps.write_scalar_summary(@resource, self.step, tag, value, typeT: dtype)
      ExecutionContext.current.add_to_collection(Graph::GraphKeys::SUMMARY_COLLECTION, result)
      result
    end

    def write(tag, value, metadata: "".b)
      value = Tensor.new(value)
      dtype ||= value.dtype

      result = RawOps.write_summary(@resource, step, value, tag, metadata, typeT: dtype)
      ExecutionContext.current.add_to_collection(Graph::GraphKeys::SUMMARY_COLLECTION, result)
      result
    end
    alias :generic :write

    def flush
      RawOps.flush_summary_writer(@resource)
    end

    def close
      RawOps.close_summary_writer(@resource)
    end
  end
end