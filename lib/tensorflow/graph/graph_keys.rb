module Tensorflow
  module Graph
    class GraphKeys
      GLOBAL_VARIABLES = "variables"
      LOCAL_VARIABLES = "local_variables"
      METRIC_VARIABLES = "metric_variables"
      MODEL_VARIABLES = "model_variables"
      TRAINABLE_VARIABLES = "trainable_variables"
      SUMMARIES = "summaries"
      QUEUE_RUNNERS = "queue_runners"
      TABLE_INITIALIZERS = "table_initializer"
      ASSET_FILEPATHS = "asset_filepaths"
      MOVING_AVERAGE_VARIABLES = "moving_average_variables"
      REGULARIZATION_LOSSES = "regularization_losses"
      CONCATENATED_VARIABLES = "concatenated_variables"
      SAVERS = "savers"
      WEIGHTS = "weights"
      BIASES = "biases"
      ACTIVATIONS = "activations"
      UPDATE_OPS = "update_ops"
      LOSSES = "losses"
      SAVEABLE_OBJECTS = "saveable_objects"
      RESOURCES = "resources"
      LOCAL_RESOURCES = "local_resources"
      TRAINABLE_RESOURCE_VARIABLES = "trainable_resource_variables"
      INIT_OP = "init_op"
      LOCAL_INIT_OP = "local_init_op"
      READY_OP = "ready_op"
      READY_FOR_LOCAL_INIT_OP = "ready_for_local_init_op"
      SUMMARY_OP = "summary_op"
      GLOBAL_STEP = "global_step"
      EVAL_STEP = "eval_step"
      COND_CONTEXT = "cond_context"
      WHILE_CONTEXT = "while_context"
      SUMMARY_COLLECTION = "_SUMMARY_V2"
      _VARIABLE_COLLECTIONS = [
          GLOBAL_VARIABLES,
          LOCAL_VARIABLES,
          METRIC_VARIABLES,
          MODEL_VARIABLES,
          TRAINABLE_VARIABLES,
          MOVING_AVERAGE_VARIABLES,
          CONCATENATED_VARIABLES,
          TRAINABLE_RESOURCE_VARIABLES,
      ]
      _STREAMING_MODEL_PORTS = "streaming_model_ports"
    end
  end
end

