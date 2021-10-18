.. _config imports:
Dynamic Imports
======================================
Since the frame work is supposed to be used as a run-platform via its own CLI interface, we provide a mechanism to load
foreign code dynamically. In particular, this allows you to load your own models and datasets such that you can use them
with the standard CLI commands.

Importing your own Code
--------------------------------------
Dynamic import of foreign code is realized via python's module import feature. In order for the framework to be able to
locate your module, you have to specify the following config entry:

.. code:: json
    {
        imports: {
            <import-name>: {
                path: <path-to-your-module>,
                module_name: <your-module-name>
            }
        },
        ...
    }
The fields described above have the following semantics:

* *<import-name>*: Unique identifier of the import.
* *<path-to-your-module>* The relative/absolute path to the folder that contains the module to be loaded. When specifying a relative path, the path will be resolved relative to the current working directory when executing a CLI command.
* *<your-module-name>* The name of the module to be loaded.

Defining your own Module
--------------------------------------
While loading your code via the dynamic import feature outlined above enables you to extend the framework with your own
ideas, it is not aware of the modules or datasets you implemented. To be able to use your own module with the CLI
commands you have to register it with the framework. This can be done by putting the following in any file that is run
upon import (e.g. at top-level in any file or in a __init__.py):

.. code:: python
  register_module(
    "<your-module-name>",
    ModuleConfig(
      <module-factory>,
      <module-type>,
      {
        "model_cls": <model-class>,
        "loss_function": <loss-function>
      }
    )
  )

The fields described above have the following semantics:

* *<your-module-name>*: Unique identifier for your module. You will be able to refer to your module in the config file using this identifier.
* *<module-factory>*: Specifies which factory the framework should use to build your module. You can eiter implement one yourself or use the GenericModuleFactory provided by the framework. The following explanation assumes you use the GenericModuleFactory. If you provide your own, the dictionary specified as the last parameter is just passed to it upon construction.
* *<module-type>*: Specifies how the framework will supply your module with training data. Valid options include MaskedTrainingModule, NextItemPredictionModule, SequenceNextItemPredictionModule and NextItemPredictionWithNegativeSampleTrainingModule.
* *<model-class>*: The class you implemented your model logic in.
* *<loss-function>* (**optional**): If your module does not have a fixed loss function, you can specify an instance here. This instance will be passed to the module by the GenericModuleFactory.

Defining your own Dataset
--------------------------------------
Similar to importing your own module, you can also load custom dataset definitions. The procedure is similar to the one
outlined above. Registering your definition is slightly different from registering a module:

.. code:: python
  register_preprocessing_config_provider(
    "<your-dataset-name>",
    PreprocessingConfigProvider(
      <your-dataset-definition-provider>,
      <default-kwargs-for-your-provider>
    )
  )

The fields described above have the following semantics:

* *<your-dataset-name>*: Unique identifier for your dataset. You will be able to refer to your dataset in the config file using this identifier.
* *<your-dataset-definition-provider>*: A function that produces a DatasetPreprocessingConfig. The framework will load values for the parameters of the function from the config file. If any of the parameters should have default values, you can specify them using the following field.
* *<default-kwargs-for-your-provider>* (**optional**): You can use this field to specify default values for the parameters of <your-dataset-definition-provider>. Parameters are matched by name and values loaded from the config file will take precedence over the values specified here.