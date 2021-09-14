from typing import Union, Tuple

import torch

from asme.core.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation, \
    ModifiedSequenceRepresentation
from asme.core.models.common.layers.layers import SequenceRepresentationLayer, ProjectionLayer
from asme.core.models.common.layers.transformer_layers import TransformerLayer, TransformerEmbedding


class SASRecTransformerComponent(SequenceRepresentationLayer):

    def __init__(self, transformer_layer: TransformerLayer):
        super().__init__()
        self.transformer_layer = transformer_layer

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence

        # pipe the embedded sequence to the transformer
        # first build the attention mask
        input_size = sequence.size()
        batch_size = input_size[0]
        sequence_length = input_size[1]

        attention_mask = torch.triu(torch.ones([sequence_length, sequence_length], device=sequence.device)) \
            .transpose(1, 0).unsqueeze(0).repeat(batch_size, 1, 1)

        if embedded_sequence.input_sequence.has_attribute("padding_mask"):
            padding_mask = embedded_sequence.input_sequence.get_attribute("padding_mask")
            attention_mask = attention_mask * padding_mask.unsqueeze(1).repeat(1, sequence_length, 1)

        attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.bool)
        representation = self.transformer_layer(sequence, attention_mask=attention_mask)

        return SequenceRepresentation(representation)


class SASRecProjectionComponent(ProjectionLayer):

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