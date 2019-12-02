require_relative "../base_test"
require 'base64'

module Tensorflow
  class IOTest < BaseTest
    def setup
      Tensorflow.execution_mode = Tensorflow::EAGER_MODE
    end

    def test_decode_base64
      message = "hello"
      encoded = Base64.urlsafe_encode64(message)
      assert_equal message, Tensorflow::IO.decode_base64(encoded).value
    end

    def test_read_file
      tempfile = Dir::Tmpname.create(['read_file', '.bin']) {}
      now = Time.now.to_i.to_s
      File.binwrite(tempfile, now)
      assert_equal(now, Tensorflow::IO.read_file(tempfile).value)
    ensure
      File.delete(tempfile)
    end

    def test_write_file
      tempfile = Dir::Tmpname.create(['write_file', '.bin']) {}
      now = Time.now.to_i.to_s
      Tensorflow::IO.write_file(tempfile, now)
      assert_equal(now, File.binread(tempfile))
    ensure
      File.delete(tempfile)
    end
  end
end