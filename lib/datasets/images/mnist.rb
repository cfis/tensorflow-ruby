$LOAD_PATH.unshift(File.expand_path('../../../lib', __dir__))

require 'tensorflow'

require_relative '../download_manager'

module Tensorflow
  module Datasets
    module Images
      class Mnist
        extend Decorator

        BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist'

        @tf.function(input_signature=[[:string]])
        def decode_image(image)
          image = IO.decode_raw(image, Tf.uint8)
          image = Tf.cast(image, Tf.float32)
          image = Tf.reshape(image, [784])
          # Normalize from [0, 255] to [0.0, 1.0]
          image / 255.0
        end

        @tf.function(input_signature=[[:string]])
        def decode_label(label)
          # tf.string -> [Tf.uint8]
          label = Tf.decode_raw(label, Tf.uint8)
          label = Tf.reshape(label, [])  # label is a scalar
          Tf.cast(label, Tf.int32)
        end

        def dataset(images_file, labels_file)
          download_manager = Tensorflow::Datasets::DownloadManager.new
          urls = ["#{BASE_URL}/#{images_file}",
                  "#{BASE_URL}/#{labels_file}"]

          resources = download_manager.download(urls)

          #images = Tensorflow::Data::FixedLengthRecordDataset.new(resources.first.path, 28 * 28, header_bytes: 16, compression_type: 'GZIP')
          images = Tensorflow::Data::FixedLengthRecordDataset.new(resources.first.path, 28 * 28, header_bytes: 16, compression_type: 'GZIP').map_func(self.decode_image)
          #labels = Tensorflow::Data::FixedLengthRecordDataset.new(labels.path, 1, header_bytes: 8, compression_type: 'GZIP').map(self.decode_image)
          #zipped = Tensorflow::Data::ZipDataset.new(images, labels)
        end

        def train_dataset
          dataset('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
        end

        def test_dataset
          dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
        end
      end
    end
  end
end