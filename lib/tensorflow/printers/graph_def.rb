require 'erubi'

module Tensorflow
  module Printers
    class GraphDef
      attr_reader :graph_def

      def initialize(graph_def)
        @graph_def = graph_def
      end

      def template
        @template ||= begin
          path = File.join(__dir__, 'graph_def.erb')
          File.read(path, :mode => 'rb')
        end
      end

      def print(io_stream=STDOUT)
        #io_stream << ERB.new(self.template, nil, trim_mode: "<>").result_with_hash(:graph_def => self.graph_def)
        raw = Erubi::Engine.new(self.template, filename: 'graph_def.erb')
        io_stream << eval(raw.src)
      end
    end
  end
end
