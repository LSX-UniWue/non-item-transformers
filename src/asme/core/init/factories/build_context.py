from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context


class BuildContext:

    def __init__(self, config: Config, context: Context):
        self.config = config
        self.context = context
        self.section = []

    def get_config(self) -> Config:
        return self.config

    def get_context(self) -> Context:
        return self.context

    def get_section(self) -> List[str]:
        return self.section

    def enter_section(self, sub_section: str) -> List[str]:
        self.section.append(sub_section)
        return self.section

    def enter_sections(self, sub_sections: List[str]) -> List[str]:
        for sub_section in sub_sections:
            self.enter_section(sub_section)

        return self.section

    def leave_section(self) -> List[str]:
        self.section.pop()
        return self.section

    def leave_sections(self, sub_sections: List[str]) -> List[str]:
        for sub_section in sub_sections:
            self.leave_section()
        return self.section

    def get_current_config_section(self) -> Config:
        return self.config.get_config(self.section)
