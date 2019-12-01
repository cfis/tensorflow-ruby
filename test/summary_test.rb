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
        self.delete_event_files
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
        self.delete_event_files
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
        self.delete_event_files
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
        self.delete_event_files
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

    def test_audio
      self.eager_and_graph do |context|
        self.delete_event_files
        writer = Summary.create_file_writer(Dir.tmpdir)

        self.evaluate(writer.initializer)
        writer.step = 1
        self.evaluate(writer.audio('audio', [[1.0]], 1.0, max_outputs: 1))
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
        assert_equal(:summary, event.what)
        assert_equal(1, event.summary.value.length)
        value = event.summary.value[0]
        assert_equal('audio/audio', value.tag)

        audio = value.audio
        assert_equal(1, audio.sample_rate)
        assert_equal(1, audio.num_channels)
        assert_equal(1, audio.length_frames)
      end
    end

    def test_histogram
      self.eager_and_graph do |context|
        self.delete_event_files
        writer = Summary.create_file_writer(Dir.tmpdir)

        self.evaluate(writer.initializer)
        writer.step = 1
        self.evaluate(writer.histogram('histogram', [1.0, 2.0, 3.0]))
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
        assert_equal(:summary, event.what)
        assert_equal(1, event.summary.value.length)
        value = event.summary.value[0]
        assert_equal('histogram', value.tag)

        histogram = value.histo
        assert_equal(1, histogram.min)
        assert_equal(3, histogram.max)
        assert_equal(6, histogram.sum)
        assert_equal(14, histogram.sum_squares)
      end
    end

    def test_image
      self.eager_and_graph do |context|
        self.delete_event_files
        writer = Summary.create_file_writer(Dir.tmpdir)

        self.evaluate(writer.initializer)
        writer.step = 1
        self.evaluate(writer.image('image', Numo::NArray[[[[1.0]]]]))
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
        assert_equal(:summary, event.what)
        assert_equal(1, event.summary.value.length)
        value = event.summary.value[0]
        assert_equal('image/image/0', value.tag)
        image = value.image
        assert_equal(1, image.width)
        assert_equal(1, image.height)

        expected = <<~BYTES
          \x89PNG\r
          \x1A
          \x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\b\x00\x00\x00\x00:~\x9BU\x00\x00\x00
          IDAT\b\x99c\xF8\x0F\x00\x01\x01\x01\x00\r\xE66\xC3\x00\x00\x00\x00IEND\xAEB`\x82
        BYTES
        assert_equal(expected.b.strip, image.encoded_image_string)
      end
    end

    def test_all_v2_summary_ops
      ops = []

      self.graph_mode do |context|
        writer = Summary.create_file_writer(Dir.tmpdir)

        # TF 2.0 summary ops
        ops << writer.write('write', 1)
        ops << writer.proto('raw_pb', '')

        # TF 1.x tf.contrib.summary ops
        writer.step = 1
        ops << writer.write('tensor', 1)
        ops << writer.scalar('scalar', 2.0)
        ops << writer.histogram('histogram', [1.0])
        ops << writer.image('image', Numo::NArray[[[[1.0]]]])
        ops << writer.audio('audio', [[1.0]], 1.0, max_outputs: 1)

        assert_equal(ops, Summary.all_v2_summary_ops)
      end
    end
  end
end