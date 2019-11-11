require 'net/http'
require 'tmpdir'
require_relative './source'

module Tensorflow
  module Datasets
    #  Manages the download and extraction of files, as well as caching.
    #
    # Downloaded files are cached under `download_dir`. The file name of downloaded
    #  files follows pattern "${sanitized_url}${content_checksum}.${ext}".
    # Eg:
    #  'cs.toronto.edu_kriz_cifar-100-pythonJDF[...]I.tar.gz'.
    #
    # While a file is being downloaded, it is placed into a directory following a
    # similar but different pattern:
    #
    #   "%{sanitized_url}${url_checksum}.tmp.${uuid}".
    #
    # When a file is downloaded, a "%{fname}s.INFO.json" file is created next to it.
    # This INFO file contains the following information:
    #
    #  {"dataset_names": ["name1", "name2"],
    #   "urls": ["http://url.of/downloaded_file"]}
    #
    # Extracted files/dirs are stored under `extract_dir`. The file name or
    # directory name is the same as the original name, prefixed with the extraction
    # method. E.g.
    #  "${extract_dir}/TAR_GZ.cs.toronto.edu_kriz_cifar-100-pythonJDF[...]I.tar.gz".
    #
    # The function members accept either plain value, or values wrapped into list
    # or dict. Giving a data structure will parallelize the downloads.
    #
    # Example of usage:
    #
    # train_dir = dl_manager.download_and_extract('https://abc.org/train.tar.gz')
    # test_dir = dl_manager.download_and_extract('https://abc.org/test.tar.gz')
    #
    # Parallel download: list -> list
    #   image_files = dl_manager.download(['https://a.org/1.jpg', 'https://a.org/2.jpg', ...])
    #
    # Parallel download: dict -> dict
    #   data_dirs = dl_manager.download_and_extract({'train': 'https://abc.org/train.zip',
    #                                                'test': 'https://abc.org/test.zip'})
    # data_dirs['train']
    # data_dirs['test']
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