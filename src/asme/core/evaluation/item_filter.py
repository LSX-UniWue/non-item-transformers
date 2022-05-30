from pathlib import Path
from asme.core.utils.ioutils import load_file_with_item_ids

class FilterPredictionItems:
    """
    Load file with selected item ids for filtering results
    """

    def __init__(self, selected_items_file: Path):
        self.selected_items = None
        if selected_items_file is not None:
            self.selected_items = load_file_with_item_ids(selected_items_file)

    def get_selected_items(self):
        return self.selected_items






