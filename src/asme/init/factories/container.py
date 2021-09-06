from typing import List

from asme.init.config import Config
from asme.init.container import Container
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.factories.common.dependencies_factory import DependenciesFactory
from asme.init.factories.features.features_factory import FeaturesFactory
from asme.init.factories.features.tokenizer_factory import TOKENIZERS_PREFIX
from asme.init.factories.data_sources.datamodule import DataModuleFactory
from asme.init.factories.modules.modules import GenericModuleFactory
from asme.init.factories.trainer import TrainerBuilderFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.losses.basket.dream.dream_loss import DreamContrastiveLoss
from asme.losses.cosrec.cosrec_loss import CosRecLoss
from asme.losses.hgn.hgn_loss import HGNLoss
from asme.losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from asme.models.basket.nnrec.nnrec_model import NNRecModel
from asme.models.bert4rec.bert4rec_model import BERT4RecModel
from asme.models.caser.caser_model import CaserModel
from asme.models.cosrec.cosrec_model import CosRecModel
from asme.models.narm.narm_model import NarmModel
from asme.models.rnn.rnn_model import RNNModel
from asme.models.sasrec.sas_rec_model import SASRecModel
from asme.models.kebert4rec.kebert4rec_model import KeBERT4RecModel
from asme.models.hgn.hgn_model import HGNModel
from asme.modules.baselines.bpr_module import BprModule
from asme.modules.baselines.markov_module import MarkovModule
from asme.modules.baselines.pop_module import PopModule
from asme.modules.baselines.session_pop_module import SessionPopModule
from asme.modules.masked_training_module import MaskedTrainingModule
from asme.modules.next_item_prediction_training_module import NextItemPredictionTrainingModule, \
    NextItemPredictionWithNegativeSampleTrainingModule
from asme.modules.sequence_next_item_prediction_training_module import SequenceNextItemPredictionTrainingModule


class ContainerFactory(ObjectFactory):
    def __init__(self):
        super().__init__()
        self.features_factory = FeaturesFactory()
        self.datamodule_factory = DataModuleFactory()
        self.dependencies = DependenciesFactory(
            [
                ConditionalFactory('type', {'kebert4rec': GenericModuleFactory(MaskedTrainingModule,
                                                                               model_cls=KeBERT4RecModel,
                                                                               loss_function=None),
                                            'bert4rec': GenericModuleFactory(MaskedTrainingModule,
                                                                             model_cls=BERT4RecModel,
                                                                             loss_function=None),
                                            'caser': GenericModuleFactory(SequenceNextItemPredictionTrainingModule,
                                                                          model_cls=CaserModel),
                                            'narm': GenericModuleFactory(NextItemPredictionTrainingModule,
                                                                         model_cls=NarmModel),
                                            'sasrec': GenericModuleFactory(SequenceNextItemPredictionTrainingModule,
                                                                           SASRecBinaryCrossEntropyLoss(),
                                                                           SASRecModel),
                                            'rnn': GenericModuleFactory(module_cls=NextItemPredictionTrainingModule,
                                                                        model_cls=RNNModel),
                                            'cosrec': GenericModuleFactory(SequenceNextItemPredictionTrainingModule,
                                                                           CosRecLoss(),
                                                                           CosRecModel),
                                            'hgn': GenericModuleFactory(SequenceNextItemPredictionTrainingModule,
                                                                        HGNLoss(),
                                                                        HGNModel),
                                            'dream': GenericModuleFactory(NextItemPredictionWithNegativeSampleTrainingModule,
                                                                          DreamContrastiveLoss,
                                                                          RNNModel),
                                            'nnrec': GenericModuleFactory(module_cls=NextItemPredictionTrainingModule,
                                                                          model_cls=NNRecModel),
                                            'pop': GenericModuleFactory(PopModule, model_cls=None),
                                            'session_pop': GenericModuleFactory(SessionPopModule, model_cls=None),
                                            'markov': GenericModuleFactory(MarkovModule, model_cls=None),
                                            'bpr': GenericModuleFactory(BprModule, model_cls=None)},
                                   config_key='module',
                                   config_path=['module']),
                TrainerBuilderFactory()
            ]
        )

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:

        datamodule_config = config.get_config(self.datamodule_factory.config_path())
        can_build_result = self.datamodule_factory.can_build(datamodule_config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = self.dependencies.can_build(config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> Container:

        # We have to build the datamodule first such that we can invoke preprocessing
        datamodule_config = config.get_config(self.datamodule_factory.config_path())
        datamodule = self.datamodule_factory.build(datamodule_config, context)
        context.set(self.datamodule_factory.config_path(), datamodule)
        # Preprocess the dataset
        datamodule.prepare_data()

        features_config = config.get_config(self.features_factory.config_path())
        meta_information = list(self.features_factory.build(features_config, context).values())
        context.set(features_config.base_path, meta_information)
        for info in meta_information:
            if info.tokenizer is not None:
                context.set([TOKENIZERS_PREFIX, info.feature_name], info.tokenizer)

        all_dependencies = self.dependencies.build(config, context)

        for key, object in all_dependencies.items():
            if isinstance(object, dict):
                for section, o in object.items():
                    context.set([key, section], o)
            else:
                context.set(key, object)

        return Container(context.as_dict())

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return ""
