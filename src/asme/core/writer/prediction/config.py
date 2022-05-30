from asme.core.init.factories.evaluation.writer.csv_multi_line import CSVMultiLineWriterFactory
from asme.core.init.factories.evaluation.writer.csv_single_line import CSVSingleLineWriterFactory
from asme.core.writer.prediction.registry import register_writer, WriterConfig

register_writer('csv-multi-line', WriterConfig(CSVMultiLineWriterFactory))
register_writer('csv-single-line', WriterConfig(CSVSingleLineWriterFactory))