require 'erubi'

module Tensorflow
  module Printers
    class Graph
      attr_reader :graph

      def initialize(graph)
        @graph = graph
      end

      def template
        @template ||= begin
          path = File.join(__dir__, 'graph.erb')
          File.read(path, :mode => 'rb')
        end
      end

      def print(io_stream=STDOUT)
        #io_stream << ERB.new(self.template, nil, trim_mode: "<>").result_with_hash(:graph => self.graph)
        raw =  Erubi::Engine.new(self.template)
        io_stream << eval(raw.src)
      end
    end
  end
end
