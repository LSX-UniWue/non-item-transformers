from asme.core.evaluation.registry import register_evaluator, EvaluationConfig
from asme.core.init.factories.evaluation.evaluator.extract_scores import ExtractScoresEvaluatorFactory
from asme.core.init.factories.evaluation.evaluator.log_input import LogInputEvaluatorFactory
from asme.core.init.factories.evaluation.evaluator.metrics import PerSampleMetricsEvaluatorFactory
from asme.core.init.factories.evaluation.evaluator.recommendation import ExtractRecommendationEvaluatorFactory
from asme.core.init.factories.evaluation.evaluator.session_id import ExtractSampleIdEvaluatorFactory
from asme.core.init.factories.evaluation.evaluator.true_target import TrueTargetEvaluatorFactory

register_evaluator('input', EvaluationConfig(LogInputEvaluatorFactory))
register_evaluator('metrics', EvaluationConfig(PerSampleMetricsEvaluatorFactory))
register_evaluator('recommendation', EvaluationConfig(ExtractRecommendationEvaluatorFactory))
register_evaluator('scores', EvaluationConfig(ExtractScoresEvaluatorFactory))
register_evaluator('sid', EvaluationConfig(ExtractSampleIdEvaluatorFactory))
register_evaluator('target', EvaluationConfig(TrueTargetEvaluatorFactory))

