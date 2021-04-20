from enum import Enum
from functools import partial
from typing import List, Callable, Any, Union, Dict

import torch
from attr import dataclass


class PadDirection(Enum):

    RIGHT = 'right'
    LEFT = 'left'


@dataclass
class PadInformation:

    pad_token_id: int
    max_seq_length: int
    max_seq_step_length: int = None


def padded_session_collate(entries_to_pad: Dict[str, PadInformation],
                           session_length_entry: str = "session",
                           pad_direction: PadDirection = PadDirection.RIGHT
                           ):
    """
    Pads sequences with a padding token to `max_length`.

    :param entries_to_pad: a list of entries in the dictionary that need to be padded.
    :param session_length_entry: the name of the entry that is used to determine individual session length.
    :param pad_direction: from where to pad the entries
    :return: a collate function that can be used to collate session data.
    """
    return partial(_padded_session_collate, entries_to_pad, session_length_entry, pad_direction)


def _padded_session_collate(entries_to_pad: Dict[str, PadInformation],
                            session_length_entry: str,
                            pad_direction: PadDirection,
                            batch):
    from torch.utils.data.dataloader import default_collate

    def pad(x: List[Any],
            generate_padding: Union[Callable[[int], Any], partial],
            padded_length: int,
            ) -> torch.Tensor:
        # truncate the session to the left to keep the last interactions
        x = x[-padded_length:]

        padding = generate_padding(len(x))

        padded_x = x + padding if pad_direction == PadDirection.RIGHT else padding + x

        return padded_x

    def _single_item_pad(length: int,
                         pad_length: int,
                         pad_token_id: int
                         ) -> List[int]:
        return [pad_token_id] * (pad_length - length)

    padded_batch = []
    for sample in batch:
        padded_sample = dict(sample)
        padded_sample["length"] = len(padded_sample[session_length_entry])

        for entry_name, value in padded_sample.items():
            value_to_convert = value
            if isinstance(value, list) and entry_name in entries_to_pad:
                pad_info = entries_to_pad[entry_name]
                max_length = pad_info.max_seq_length
                padding_token_id = pad_info.pad_token_id

                if isinstance(value[0], list):
                    max_seq_step_length = pad_info.max_seq_step_length
                    # first pad entries in the list to the maximum seq step length
                    padded_entries = [
                        pad(value_entry, partial(_single_item_pad, pad_token_id=padding_token_id, pad_length=max_seq_step_length), max_seq_step_length) for value_entry in value
                    ]

                    value_to_convert = pad(padded_entries, lambda length: [[padding_token_id] * max_seq_step_length] * (max_length - length), max_length)
                else:
                    value_to_convert = pad(value, partial(_single_item_pad, pad_token_id=padding_token_id, pad_length=max_length), max_length)

            padded_sample[entry_name] = torch.as_tensor(value_to_convert)

        padded_batch.append(padded_sample)

    collated_batch = default_collate(padded_batch)
    return collated_batch
