from asme.models.common.layers.data.sequence import SequenceRepresentation, ModifiedSequenceRepresentation
from asme.models.common.layers.layers import SequenceRepresentationModifierLayer
from torch import nn


class FFNSequenceRepresentationModifierComponent(SequenceRepresentationModifierLayer):
    """
    layer that applies a linear layer and a activation function
    """
    def __init__(self,
                 feature_size: int
                 ):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.GELU(),
            nn.LayerNorm(feature_size)
        )

    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        transformation = self.transform(sequence_representation.encoded_sequence)
        return ModifiedSequenceRepresentation(transformation)
