class Array
  # Helper method to make writing tests easier.  Allows this to work:
  #
  #   assert_equal([1,2], Numo::Array[1,2])
  #
  # Versus having to do this:
  #   assert_equal([1,2], Numo::Array[1,2].to_a)

  alias :original_equals :==
  def ==(other)
    if other.kind_of?(Numo::NArray)
      self.eql?(other.to_a)
    else
      original_equals(other)
    end
  end
end