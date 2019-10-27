require 'set'

module Tensorflow
  class NameScope
    attr_reader :stack, :names

    def initialize(name: nil, reuse: nil, initializer: nil)
      @stack = Array.new
      @names = Set.new
    end

    def name_scope(base_name)
      name = self.unique_name(base_name)
      stack.push(name)

      begin
        yield current_scope if block_given?
      ensure
        stack.pop
      end
    end

    def current_scope
      if self.stack.last.nil?
        nil
      else
        self.stack.join("/")
      end
    end

    def scoped_name(name)
      base_name = case
                    when self.stack.empty?
                      name
                    when self.stack.last.nil?
                      name
                    else
                      "#{self.current_scope}/#{name}"
                  end

      self.unique_name(base_name)
    end

    def unique_name(name)
      return nil unless name

      i = 0
      check_name = name
      while self.names.include?(check_name.downcase)
        i += 1
        check_name = "#{name}_#{i}"
      end
      self.names << check_name.downcase unless check_name.nil?
      check_name
    end
  end
end
