$LOAD_PATH.unshift(File.expand_path('../../../lib', __dir__))

require 'tensorflow'

require_relative '../download_manager'

module Tensorflow
  module Datasets
    module Images
      class Mnist
        BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist'

        def decode_image(image)
          # Normalize from [0, 255] to [0.0, 1.0]
          image = IO.decode_raw(image, tf.uint8)
          image = tf.cast(image, tf.float32)
          image = tf.reshape(image, [784])
          return image / 255.0
        end

        def dataset(images_file, labels_file)
          download_manager = Tensorflow::Datasets::DownloadManager.new
          urls = ["#{BASE_URL}/#{images_file}",
                  "#{BASE_URL}/#{labels_file}"]

          resources = download_manager.download(urls)

          images = Tensorflow::Data::FixedLengthRecordDataset.new(images.path, 28 * 28, header_bytes: 16, compression_type: 'GZIP')
          labels = Tensorflow::Data::FixedLengthRecordDataset.new(labels.path, 1, header_bytes: 8, compression_type: 'GZIP')
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