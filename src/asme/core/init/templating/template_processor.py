from abc import abstractmethod
from typing import Dict, Any


class TemplateProcessor:

    @abstractmethod
    def can_modify(self,
                   config: Dict[str, Any]
                   ) -> bool:
        """
        :param config:
        :return: true iff the template processor can modify the config dict
        """
        pass

    @abstractmethod
    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        make all the necessary config modifications
        :param config:
        :return: the modified config
        """
        pass
