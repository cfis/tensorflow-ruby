# encoding: UTF-8

module Kernel
  def Boolean(value)
    # Rails converts true/false to 't' and 'f' in this case
    # because it does not have data dictionary information for
    # these fields and doesn't seem to be able to figure it
    # out from the query results.
    if not value
      false
    elsif value.to_s.match(/^(t|true|1|yes|y)$/i)
      true
    else
      false
    end
  end
end

class TrueClass
  def to_i() 1; end
end

class FalseClass
  def to_i() 0; end
end
