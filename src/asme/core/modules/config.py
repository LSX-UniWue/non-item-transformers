from asme.core.init.factories.modules.modules import GenericModuleFactory
from asme.core.losses.basket.dream.dream_loss import DreamContrastiveLoss
from asme.core.losses.cosrec.cosrec_loss import CosRecLoss
from asme.core.losses.hgn.hgn_loss import HGNLoss
from asme.core.losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss, SASRecFullSequenceCrossEntropyLoss
from asme.core.models.basket.nnrec.nnrec_model import NNRecModel
from asme.core.models.bert4rec.bert4rec_model import BERT4RecModel
from asme.core.models.caser.caser_model import CaserModel
from asme.core.models.cosrec.cosrec_model import CosRecModel
from asme.core.models.hgn.hgn_model import HGNModel
from asme.core.models.content_bert4rec.content_bert4rec_model import ContentBERT4RecModel
from asme.core.models.narm.narm_model import NarmModel
from asme.core.models.non_item_models.non_item_bert4rec_model import NonItemBERT4RecModel
from asme.core.models.non_item_models.non_item_sasrec_model import NonItemSASRecModel
from asme.core.models.rnn.rnn_model import RNNModel
from asme.core.models.sasrec.sasrec_model import SASRecModel
from asme.core.models.content_sasrec.content_sasrec_model import ContentSASRecModel
from asme.core.modules.baselines.bpr_module import BprModule
from asme.core.modules.baselines.genre_pop_module import GenrePopModule
from asme.core.modules.baselines.markov_module import MarkovModule
from asme.core.modules.baselines.pop_module import PopModule
from asme.core.modules.baselines.session_pop_module import SessionPopModule
from asme.core.modules.masked_training_module import MaskedTrainingModule
from asme.core.modules.non_items.next_item_in_sequence_prediction_training_module import \
    NextItemInSequencePredictionTrainingModule
# from asme.core.modules.non_items.non_item_masked_training_module import NonItemMaskedTrainingModule
from asme.core.modules.non_items.non_item_masked_training_module import NonItemMaskedTrainingModule
from asme.core.modules.prepend_content.user_masked_training_module import SequenceContentMaskedTrainingModule
from asme.core.modules.prepend_content.user_next_item_prediction_training_module import UserNextItemPredictionTrainingModule
from asme.core.modules.next_item_prediction_training_module import NextItemPredictionTrainingModule, \
    NextItemPredictionWithNegativeSampleTrainingModule
from asme.core.modules.registry import register_module, ModuleConfig
from asme.core.modules.sequence_next_item_prediction_training_module import SequenceNextItemPredictionTrainingModule

register_module('content-bert4rec', ModuleConfig(GenericModuleFactory, SequenceContentMaskedTrainingModule, {
    "model_cls": ContentBERT4RecModel}))

register_module('bert4rec', ModuleConfig(GenericModuleFactory, MaskedTrainingModule, {
    "model_cls": BERT4RecModel}))

register_module("caser", ModuleConfig(GenericModuleFactory, SequenceNextItemPredictionTrainingModule, {
    "model_cls": CaserModel}))

register_module("narm", ModuleConfig(GenericModuleFactory, NextItemPredictionTrainingModule, {
    "model_cls": NarmModel}))

register_module("sasrec-neg", ModuleConfig(GenericModuleFactory, SequenceNextItemPredictionTrainingModule, {
    "model_cls": SASRecModel,
    "loss_function": SASRecBinaryCrossEntropyLoss()}))

register_module("content-sasrec-full", ModuleConfig(GenericModuleFactory, UserNextItemPredictionTrainingModule, {
    "model_cls": ContentSASRecModel,
    "loss_function": SASRecFullSequenceCrossEntropyLoss}))

register_module("non-items-sasrec-cross", ModuleConfig(GenericModuleFactory, NextItemInSequencePredictionTrainingModule, {
    "model_cls": NonItemSASRecModel}))

register_module("non-items-bert4rec", ModuleConfig(GenericModuleFactory, NonItemMaskedTrainingModule, {
    "model_cls": NonItemBERT4RecModel}))

register_module("sasrec-cross", ModuleConfig(GenericModuleFactory, NextItemPredictionTrainingModule, {
    "model_cls": SASRecModel,
    "loss_function": SASRecFullSequenceCrossEntropyLoss}))

register_module("rnn", ModuleConfig(GenericModuleFactory, NextItemPredictionTrainingModule, {
    "model_cls": RNNModel}))

register_module("cosrec", ModuleConfig(GenericModuleFactory, SequenceNextItemPredictionTrainingModule, {
    "model_cls": CosRecModel,
    "loss_function": CosRecLoss()}))

register_module("hgn", ModuleConfig(GenericModuleFactory, SequenceNextItemPredictionTrainingModule, {
    "model_cls": HGNModel,
    "loss_function": HGNLoss()}))

register_module("dream",
                ModuleConfig(GenericModuleFactory, NextItemPredictionWithNegativeSampleTrainingModule, {
                    "model_cls": RNNModel,
                    "loss_function": DreamContrastiveLoss}))

register_module("nnrec", ModuleConfig(GenericModuleFactory, NextItemPredictionTrainingModule, {
    "model_cls": NNRecModel}))

register_module("pop", ModuleConfig(GenericModuleFactory, PopModule))

register_module("attribute_pop", ModuleConfig(GenericModuleFactory, GenrePopModule))

register_module("session_pop", ModuleConfig(GenericModuleFactory, SessionPopModule))

register_module("markov", ModuleConfig(GenericModuleFactory, MarkovModule))

register_module("bpr", ModuleConfig(GenericModuleFactory, BprModule))
