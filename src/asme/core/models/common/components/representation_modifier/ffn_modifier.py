from asme.core.models.common.layers.data.sequence import SequenceRepresentation, ModifiedSequenceRepresentation
from asme.core.models.common.layers.layers import SequenceRepresentationModifierLayer
from torch import nn

from asme.core.utils.hyperparameter_utils import save_hyperparameters


class FFNSequenceRepresentationModifierComponent(SequenceRepresentationModifierLayer):
    """
    layer that applies a linear layer and a activation function
    """

    @save_hyperparameters
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
