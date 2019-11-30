module Tensorflow
  class Summary
    def self.create_file_writer(logdir, max_queue: 10, flush_millis: 120_000, filename_suffix: '.v2', name: nil)
      ResourceSummaryWriter.new(shared_name: name) do |writer|
        RawOps.create_summary_file_writer(writer, logdir, max_queue, flush_millis, filename_suffix)
      end
    end

    def self.all_v2_summary_ops
      ExecutionContext.current.get_collection_ref(Graph::GraphKeys::SUMMARY_COLLECTION)
    end
  end
end