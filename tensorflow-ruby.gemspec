require_relative "lib/tensorflow/version"

Gem::Specification.new do |spec|
  spec.name          = "tensorflow-ruby"
  spec.version       = Tensorflow::VERSION
  spec.summary       = "Tensorflow bindings for ruby"
  spec.homepage      = "https://github.com/cfis/tensorflow-ruby"
  spec.license       = "MIT"

  spec.author        = "Charlie Savage"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.4"

  spec.add_dependency "erubi"
  spec.add_dependency "ffi"
  spec.add_dependency "numo-narray"
  spec.add_dependency "google-protobuf", ">=3.11.0"
  spec.add_dependency "npy"

  spec.add_development_dependency "bundler"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "minitest", ">= 5.10"
end