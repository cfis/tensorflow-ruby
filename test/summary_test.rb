require_relative "base_test"
require "pathname"

module Tensorflow
  class SummaryTest < BaseTest
    EVENT_FILE_GLOB = ('events.out.*v2')

    def event_files
      Pathname(Dir.tmpdir).glob(EVENT_FILE_GLOB)
    end

    def read_events(context)
      dataset = Data::TfRecordDataset.new(self.event_files.map(&:to_path))
      records = self.evaluate(dataset)
      records.map do |record|
        Tensorflow::Event.decode(record)
      end
    end

    def delete_event_files
      self.event_files.each(&:delete)
    end

    def setup
      delete_event_files
    end

    def test_write
      self.eager_and_graph do |context|
        writer = Summary.create_file_writer(Dir.tmpdir)
        writer.step = 12
        write_op = writer.write('tag', 42)
        writer.flush

        dataset = Data::TfRecordDataset.new(self.event_files.map(&:to_path))
        records = self.evaluate(dataset)
        events = records.map do |record|
          Tensorflow::Event.decode(record)
        end

        assert_equal(2, events.length)

        events = read_events(context)
        event = events[0]
        assert_equal(0, event.step)
        assert_equal(:file_version, event.what)
        assert_equal('brain.Event:2', event.file_version)

        event = events[1]
        assert_equal(12, event.step)
        assert_equal(:summary, event.what)
        assert_equal(1, event.summary.value.length)
        value = event.summary.value[0]
        assert_equal('tag', value.tag)
        tensor = Tensor.from_proto(value.tensor)
        assert_equal([], tensor.shape)
        assert_equal(42, tensor.value)
      end
    end

    def test_write_metadata
      metadata = Tensorflow::SummaryMetadata.new
      metadata.plugin_data = SummaryMetadata::PluginData.new
      metadata.plugin_data.plugin_name = 'foo'
      encoded = Tensorflow::SummaryMetadata.encode(metadata)

      self.eager_and_graph do |context|
        writer = Summary.create_file_writer(Dir.tmpdir)
        writer.write('obj', 0, metadata: metadata)
        writer.write('bytes', 0, metadata: encoded)
        m = Tensorflow.constant(encoded)
        writer.write('string_tensor', 0, metadata: m)
        writer.flush

        events = read_events(context)
        assert_equal(4, events.length)
        assert_equal(metadata, events[1].summary.value[0].metadata)
        assert_equal(metadata, events[2].summary.value[0].metadata)
        assert_equal(metadata, events[3].summary.value[0].metadata)
      end
    end

    def test_write_narray
      self.eager_and_graph do |context|
        writer = Summary.create_file_writer(Dir.tmpdir)
        writer.step = 2
        write_op = writer.write('tag', [[1, 2], [3, 4]])
        writer.flush

        events = read_events(context)
        assert_equal(2, events.length)

        tensor = Tensor.from_proto(events[1].summary.value[0].tensor)
        assert_equal([[1, 2], [3, 4]], tensor.value)
      end
    end

    def test_write_using_default_step
      self.eager_and_graph do |context|
        writer = Summary.create_file_writer(Dir.tmpdir)

        writer.step = 1
        writer.write('tag', 1.0)

        writer.step = 2
        writer.write('tag', 1.0)

        mystep = Variable.new(10, dtype: :int64)
        writer.step = mystep
        writer.write('tag', 1.0)

        mystep.assign_add(1)
        writer.write('tag', 1.0)
        writer.flush

        events = read_events(context)
        assert_equal(5, events.length)

        assert_equal(1, events[1].step)
        assert_equal(2, events[2].step)
        assert_equal(10, events[3].step)
        assert_equal(11, events[4].step)
      end
    end

    def test_graph
      self.graph_mode do |context|
        writer = Summary.create_file_writer(Dir.tmpdir)

        self.evaluate(writer.initializer)
        self.evaluate(writer.graph(context))
        self.evaluate(writer.flush)

        dataset = Data::TfRecordDataset.new(self.event_files.map(&:to_path))
        records = self.evaluate(dataset)
        events = records.map do |record|
          Tensorflow::Event.decode(record)
        end

        assert_equal(2, events.length)

        events = read_events(context)
        event = events[0]
        assert_equal(0, event.step)
        assert_equal(:file_version, event.what)
        assert_equal('brain.Event:2', event.file_version)

        event = events[1]
        assert_equal(1, event.step)
        assert_equal(:graph_def, event.what)

        refute_nil(event.graph_def)
      end
    end
  end
end