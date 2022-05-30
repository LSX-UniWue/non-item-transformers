from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.writer.prediction.batch_prediction_writer import CSVMultiLineWriter


class CSVMultiLineWriterFactory(ObjectFactory):
    """
    Factory for the CSVMultiLineWriter
    """

    def can_build(self,  build_context: BuildContext) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,  build_context: BuildContext) -> CSVMultiLineWriter:
        evaluators = build_context.get_context().get("evaluation")["evaluators"]
        return CSVMultiLineWriter(evaluators)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return ["writer"]

    def config_key(self) -> str:
        return 'evaluation'
