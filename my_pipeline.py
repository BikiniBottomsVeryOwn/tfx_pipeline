import os
from typing import Optional, Text, List
from absl import logging
from ml_metadata.proto import metadata_store_pb2
import tfx.v1 as tfx
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import Tuner
from tfx.components import Evaluator
from tfx.components import InfraValidator
from tfx.components import Pusher

import tensorflow_model_analysis as tfma
from models import preprocessing

DATA_PATH = os.path.join('.', 'data')
MODEL_PATH = 'models/model.py'
PIPELINE_ROOT = os.path.join('.', 'my_pipeline_output')
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
PIPELINE_NAME = 'my_pipeline'
METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
PREPROCESSING_FN = 'models/preprocessing.py'
MAKE_WARMUP = False
ENABLE_CACHE = True
TUNER_PATH = 'models/model.py'
ACCURACY_THRESHOLD = 0.5

def create_pipeline(
  pipeline_name: Text,
  pipeline_root:Text,
  data_path: Text,
  preprocessing_fn: Text,
  model_path: Text,
  tuner_path: Text,
  accuracy_threshold: float,
  make_warmup: bool,
  serving_model_dir:Text,
  enable_cache: bool,
  metadata_connection_config: Optional[
    metadata_store_pb2.ConnectionConfig] = None,
  beam_pipeline_args: Optional[List[Text]] = None,
):
  components = []

  example_gen = CsvExampleGen(input_base=data_path)
  components.append(example_gen)

  stats_gen = StatisticsGen(
      examples=example_gen.outputs['examples']
      )
  components.append(stats_gen)

  schema_gen = SchemaGen(
    statistics=stats_gen.outputs['statistics'])
  components.append(schema_gen)

  transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=preprocessing_fn)
  
  components.append(transform)


  tuner = Tuner(
    module_file=tuner_path,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=tfx.proto.TrainArgs(num_steps=20),
    eval_args=tfx.proto.EvalArgs(num_steps=5))

  components.append(tuner)
      
  trainer = Trainer(
    module_file=model_path,  # Contains `run_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=tfx.proto.TrainArgs(num_steps=100),
    eval_args=tfx.proto.EvalArgs(num_steps=5))
    

  components.append(trainer)

  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

  components.append(model_resolver)

  eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(
            signature_name='serving_default',
            label_key='species_xf',
            preprocessing_function_names=['transform_features'])
    ],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(
                class_name='SparseCategoricalAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': accuracy_threshold}),
                    # Change threshold will be ignored if there is no
                    # baseline model resolved from MLMD (first run).
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': -1e-10})))
        ])
    ])

  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  components.append(evaluator)

  infra_validator = InfraValidator(
      model=trainer.outputs['model'],
      examples=example_gen.outputs['examples'],
      serving_spec=tfx.proto.ServingSpec(
          tensorflow_serving=tfx.proto.TensorFlowServing(  # Using TF Serving.
              tags=['latest']
          ),
          local_docker=tfx.proto.LocalDockerConfig(),  # Running on local docker.
      ),
      validation_spec=tfx.proto.ValidationSpec(
          max_loading_time_seconds=60,
          num_tries=5,
      ),
      request_spec=tfx.proto.RequestSpec(
          tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(),
          num_examples=1,
      )
)


  components.append(infra_validator)



  pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(
      filesystem=tfx.proto.PushDestination.Filesystem(
          base_directory=serving_model_dir)
    )
  )
  components.append(pusher)


  return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args, 
    )

def run_pipeline():
  my_pipeline = create_pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      data_path=DATA_PATH,
      accuracy_threshold=ACCURACY_THRESHOLD,
      preprocessing_fn=PREPROCESSING_FN,
      model_path=MODEL_PATH,
      tuner_path=TUNER_PATH,
      serving_model_dir=SERVING_MODEL_DIR,
      make_warmup=MAKE_WARMUP,
      enable_cache=ENABLE_CACHE,
      metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
      )

  tfx.orchestration.LocalDagRunner().run(my_pipeline)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run_pipeline()





#https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py