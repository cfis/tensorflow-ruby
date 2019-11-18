require "bundler/setup"
Bundler.require(:default)

require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false
end

class RawOpHelper
  # based on ActiveSupport underscore
  def self.underscore(str)
    str.gsub(/([A-Z]+)([A-Z][a-z])/,'\1_\2').gsub(/([a-z\d])([A-Z])/,'\1_\2').downcase
  end

  def self.check_name(name)
    # start and stop choosen as they are used for some operations
    case name
      when "begin"
        "start"
      when "end"
        "stop"
      else
        name
    end
  end

  def self.check_attribute_name(attr_def)
    name = self.check_name(attr_def.name)
    if name == 'T'
      'typeT'
    else
      name.downcase
    end
  end

  def self.convert_type(type)
    # Will be something like DT_INT32
    type[3..].downcase.to_sym
  end

  def self.process_attribute(attr_def)
    name = self.check_attribute_name(attr_def)
    default_value = if attr_def.default_value.nil? && !attr_def.allowed_values
                      'nil'
                    elsif attr_def.default_value.nil? && attr_def.allowed_values
                      'nil'
                      # case attr_def.type
                      #   when 'type', 'list(type)'
                      #     values = attr_def.allowed_values.list.type
                      #     value = if values.include?(:DT_INT32)
                      #               'DT_INT32'
                      #             else
                      #               values.first
                      #             end
                      #     ":#{self.convert_type(value)}"
                      #   when 'string'
                      #     values = attr_def.allowed_values.list.s
                      #     values.first
                      #   else
                      #     raise('Unsupported allowed value')
                      # end
                    else
                      case attr_def.default_value.value
                        when :s
                          "\"#{attr_def.default_value['s']}\""
                        when :list
                          []
                        when :shape
                          []
                        when :tensor
                          []
                        when :type
                          value = self.convert_type(attr_def.default_value[attr_def.default_value.value.to_s])
                          ":#{value.downcase}"
                        else
                          attr_def.default_value[attr_def.default_value.value.to_s]
                      end
                    end

    "#{name}: #{default_value}"
  end
end

desc 'Generate raw_ops.rb from Tensorflow operations'
task :generate_ops do
  require 'erb'
  template = File.read('lib/tensorflow/ops/raw_ops.rb.erb', mode: 'rb')
  content = ERB.new(template, nil, trim_mode: "%-").result(binding)
  File.write("lib/tensorflow/ops/raw_ops.rb", content)
end