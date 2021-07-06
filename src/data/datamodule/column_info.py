from dataclasses import dataclass
from typing import Optional


@dataclass
class ColumnInfo:
    columnName: str
    delimiter: Optional[str] = None
