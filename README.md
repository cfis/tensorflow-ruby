# Tensorflow

Ruby bindings for [Tensorflow](https://github.com/tensorflow/tensorflow).

## Installation

First you'll need to install Tensorflow 2. You can either do a full [installation](https://www.tensorflow.org/install
) or just install [Tensorflow for C](https://www.tensorflow.org/install/lang_c). In both cases, you'll need to make 
sure the tensorflow library is on the system PATH so the Ruby bindings can load it.

Next install the gem:

```ruby
gem install 'tensorflow-ruby'
```

## Overview

The Ruby bindings attempt to mimic the [Python API](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf). Just like the
Python bindings, Ruby defaults to [eager](https://www.tensorflow.org/guide/eager) execution mode.
 
Currently the bindings support:

* Ability to call all Tensorflow operations
* Eager execution mode
* Graph execution mode
* Importing and exporting computation graphs
* Creating and editing computation graphs
* Creating custom operators
* Gradients

Unimplemented:

* Autodiff
* Keras API
* Training models
* Conversion of Ruby methods to graph functions

## Constants

```ruby
a = Tf.constant([1, 2, 3])
b = Tf.constant([4, 5, 6])
a + b
```

## Variables

```ruby
v = Tf::Variable.new(0.0)
w = v + 1
```

## Math

```ruby
Tf::Math.abs([-1, -2])
Tf::Math.sqrt([1.0, 4.0, 9.0])
```

## Data::Dataset

```ruby
# load
train_dataset = Tf::Data::Dataset.from_tensor_slices([train_examples, train_labels])
test_dataset = Tf::Data::Dataset.from_tensor_slices([test_examples, test_labels])

# shuffle and batch
train_dataset = train_dataset.shuffle(100).batch(32)
test_dataset = test_dataset.batch(32)

# iterate
train_dataset.each do |examples, labels|
  # ...
end
```
## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/cfis/tensorflow-ruby/issues)
- Fix bugs and [submit pull requests](https://github.com/cfis/tensorflow-ruby/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development and testing:

```sh
git clone https://github.com/cfis/tensorflow-ruby.git
cd tensorflow-ruby
bundle install
rake test
```
