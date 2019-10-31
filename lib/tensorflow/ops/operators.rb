module Tensorflow
  module Operators
    def +(other)
      Math.add(self, other)
    end

    def -(other)
      Math.subtract(self, other)
    end

    def *(other)
      Math.multiply(self, other)
    end

    def **(other)
      Math.pow(self, other)
    end

    def /(other)
      Math.divide(self, other)
    end

    def %(other)
      Math.floormod(self, other)
    end

    def -@
      Math.negative(self)
    end
  end
end