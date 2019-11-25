module Tensorflow
  module Operators
    def +(other)
      Math.add(self, other, dtype: self.dtype)
    end

    def -(other)
      Math.subtract(self, other, dtype: self.dtype)
    end

    def *(other)
      Math.multiply(self, other, dtype: self.dtype)
    end

    def **(other)
      Math.pow(self, other, dtype: self.dtype)
    end

    def /(other)
      Math.divide(self, other, dtype: self.dtype)
    end

    def %(other)
      Math.floormod(self, other, dtype: self.dtype)
    end

    def -@
      Math.negative(self)
    end
  end
end