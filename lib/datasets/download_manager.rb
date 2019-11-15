require 'net/http'
require 'tmpdir'
require_relative './resource'

module Tensorflow
  module Datasets
    class DownloadManager
      attr_reader :uri, :dir

      def initialize(dir = Dir.tmpdir)
        @dir = dir
      end

      def download(urls)
        resources = Array(urls).flatten.map {|url| Resource.new(url)}
        resources.each do |resource|
          self.download_resource(resource)
        end
        resources
      end

      def download_resource(resource)
        resource.path = File.join(self.dir, resource.filename)
        return if File.exist?(resource.path)

        STDOUT << "Downloading #{resource.uri}" << "\n"
        http = Net::HTTP.new(resource.uri.host, resource.uri.port)
        http.use_ssl = resource.uri.is_a?(URI::HTTPS)

        request = Net::HTTP::Get.new(resource.uri.request_uri)

        http.start do |http|
          http.request(request) do |response|
            file_size = response['content-length'].to_f
            bytes = 0

            File.open(resource.path, 'wb') do |file|
              response.read_body do |chunk|
                file.write(chunk)
                bytes += chunk.size
                #STDOUT << "#{(bytes / file_size * 100).to_i}%\r"
              end
            end
          end
        end
      end
    end
  end
end