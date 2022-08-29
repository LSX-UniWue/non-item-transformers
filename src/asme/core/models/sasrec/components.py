from typing import Union, Tuple, Dict, Any

import torch

from asme.core.models.common.layers.data.sequence import ModifiedSequenceRepresentation, SequenceRepresentation
from asme.core.models.common.layers.layers import ProjectionLayer, SequenceRepresentationModifierLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.kebert4rec.components import _build_embedding_type
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from torch import nn


class SASRecProjectionComponent(ProjectionLayer):

    @save_hyperparameters
    def __init__(self, embedding: TransformerEmbedding):
        super().__init__()
        self.embedding = embedding

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        input_sequence = modified_sequence_representation.input_sequence

        positive_samples = input_sequence.get_attribute("positive_samples")
        negative_samples = input_sequence.get_attribute("negative_samples")

        padding_mask = input_sequence.padding_mask
        transformer_output = modified_sequence_representation.modified_encoded_sequence
        # TODO (AD) We should switch between training / prediction based on Module state, see: torch.nn.Module/train
        # when training the model we multiply the seq embedding with the positive and negative items
        if negative_samples is not None:
            emb_pos_items = self.embedding.get_item_embedding(positive_samples)  # (N, H)
            emb_neg_items = self.embedding.get_item_embedding(negative_samples)  # (N, H)

            pos_output = emb_pos_items * transformer_output  # (N, S, H)
            neg_output = emb_neg_items * transformer_output  # (N, S, H)

            pos_output = torch.sum(pos_output, -1)  # (N, S)
            neg_output = torch.sum(neg_output, -1)  # (N, S)

            return pos_output, neg_output

        # inference step (I is the number of positive items to test)
        # embeddings of pos_items
        item_embeddings = self.embedding.get_item_embedding(positive_samples, flatten=False)  # (N, I, H)

        # we use "advanced" indexing to slice the right elements from the transformer output
        batch_size = transformer_output.size()[0]
        batch_index = torch.arange(0, batch_size)

        # calculate indices from the padding mask
        seq_index = padding_mask.sum(-1) - 1
        transformer_last_pos_output = transformer_output[batch_index, seq_index]  # (N, H)

        # now matmul it with the item embeddings
        logits = item_embeddings.matmul(transformer_last_pos_output.unsqueeze(-1))

        return logits.squeeze(-1)

class PostFusionIdentitySequenceRepresentationModifierLayer(SequenceRepresentationModifierLayer):

    """ a SequenceRepresentationModifierLayer that does nothing with the sequence representation """

    @save_hyperparameters
    def __init__(self,
                 feature_size: int,
                 postfusion_attributes: Dict[str, Dict[str, Any]],
                 additional_attributes_tokenizer: Dict[str, Tokenizer],
                 merge_function: str = "add"
                 ):
        super().__init__()

        self.merge_function = merge_function

        postfusion_attribute_embeddings = {}
        for attribute_name, attribute_infos in postfusion_attributes.items():
            embedding_type = attribute_infos['embedding_type']
            vocab_size = len(additional_attributes_tokenizer["tokenizers." + attribute_name])
            postfusion_attribute_embeddings[attribute_name] = _build_embedding_type(embedding_type=embedding_type,
                                                                                    vocab_size=vocab_size,
                                                                                    hidden_size=feature_size)
        self.postfusion_attribute_embeddings = nn.ModuleDict(postfusion_attribute_embeddings)


    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        postfusion_embedded_sequence = sequence_representation.encoded_sequence

        #Sum attribute embeddings
        context_embeddings = None
        for input_key, module in self.postfusion_attribute_embeddings.items():
            additional_metadata = sequence_representation.input_sequence.get_attribute(input_key)
            pf_attribute = module(additional_metadata)
            if context_embeddings is None:
                context_embeddings = pf_attribute
            else:
                context_embeddings += pf_attribute

        if self.merge_function == "add":
            postfusion_embedded_sequence +=context_embeddings
        if self.merge_function == "multiply":
            postfusion_embedded_sequence *= context_embeddings

        return ModifiedSequenceRepresentation(postfusion_embedded_sequence)



