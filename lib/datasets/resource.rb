require 'uri'

module Tensorflow
  module Datasets
    class Resource
      attr_reader :uri
      attr_accessor :path

      def initialize(url)
        @uri = url.is_a?(URI) ? url : URI.parse(url)
      end

      def filename
        File.basename(self.uri.path)
      end
    end
  end
end

