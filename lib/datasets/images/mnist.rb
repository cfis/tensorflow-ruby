$LOAD_PATH.unshift(File.expand_path('../../..', __FILE__))

require 'tensorflow'

require_relative '../download_manager'

module Tensorflow
  module Datasets
    module Images
      class Mnist
        BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist'

        def dataset(images_file, labels_file)
          download_manager = Tensorflow::Datasets::DownloadManager.new
          urls = ["#{BASE_URL}/#{images_file}",
                  "#{BASE_URL}/#{labels_file}"]

          download_manager.download(urls)
        end

        def train_dataset
          images, labels = dataset('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
          images = Tensorflow::Data::FixedLengthRecordDataset.new(images.path, 28 * 28, header_bytes: 16, compression_type: 'GZIP')
          labels = Tensorflow::Data::FixedLengthRecordDataset.new(labels.path, 1, header_bytes: 8, compression_type: 'GZIP')
          zipped = Tensorflow::Data::ZipDataset.new(images, labels)
        end
      end
    end
  end
end

mnist = Tensorflow::Datasets::Images::Mnist.new
mnist.train_dataset