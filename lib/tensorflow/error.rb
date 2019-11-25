module Tensorflow
  module Error
    class AbortedError < StandardError
    end 

    class AlreadyExistsError < StandardError
    end

    class CancelledError < StandardError
    end

    class DataLossError < StandardError
    end

    class DeadlineExceededError < StandardError
    end

    class FailedPreconditionError < StandardError
    end

    class InternalError < StandardError
    end

    class InvalidArgumentError < StandardError
    end

    class NotFoundError < StandardError
    end

    class OpError < StandardError
    end

    class OutOfRangeError < StandardError
    end

    class PermissionDeniedError < StandardError
    end

    class ResourceExhaustedError < StandardError
    end

    class UnauthenticatedError < StandardError
    end

    class UnavailableError < StandardError
    end

    class UnimplementedError < StandardError
    end

    class UnknownError < StandardError
    end
  end
end
