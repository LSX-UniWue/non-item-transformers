from torch import nn

ACTIVATION_FUNCTION_KEY_RELU = 'relu'


def get_activation_layer(activation_fn_name: str) -> nn.Module:
    """
    :param activation_fn_name: the name of the activation function
    :return: the torch layer for the specified layer name
    """
    return {
        'identity': nn.Identity(),
        ACTIVATION_FUNCTION_KEY_RELU: nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'glu': nn.GLU()
    }[activation_fn_name]
