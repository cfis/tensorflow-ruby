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
          label = Tf::IO.decode_raw(label, Tf.uint8)
          label = Tf.reshape(label, [])  # label is a scalar
          Tf.cast(label, Tf.int32)
        end

        def dataset(images_file, labels_file)
          download_manager = Datasets::DownloadManager.new
          urls = ["#{BASE_URL}/#{images_file}",
                  "#{BASE_URL}/#{labels_file}"]

          resources = download_manager.download(urls)

          images = Data::FixedLengthRecordDataset.new(resources.first.path, 28 * 28, header_bytes: 16, compression_type: 'GZIP').map_func(self.decode_image)
          labels = Data::FixedLengthRecordDataset.new(resources.last.path, 1, header_bytes: 8, compression_type: 'GZIP').map_func(self.decode_label)
          Data::ZipDataset.new(images, labels)
        end

        def train
          dataset('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
        end

        def test
          dataset('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
        end
      end
    end
  end
end