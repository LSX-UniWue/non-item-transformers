from typing import Any, Dict

import optuna
import pytest
from optuna import Trial

from asme.core.init.templating.search.processor import SearchTemplateProcessor
from asme.core.init.templating.search.resolver import OptunaParameterResolver


@pytest.fixture
def base_processor():
    study = optuna.create_study(optuna.storages.InMemoryStorage())
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    return SearchTemplateProcessor(OptunaParameterResolver(trial))


@pytest.fixture
def template():
    return {
        "model":
            {
                "batch_size": 32,
                "layer_size": {
                    "hyper_opt": {
                        "suggest": "int",
                        "params": {
                            "low": 8,
                            "high": 64
                        }
                    }
                }
            },
        "trainer": {
            "learning_rate": {
                "hyper_opt": {
                    "suggest": "loguniform",
                    "params": {
                        "low": 0.00001,
                        "high": 0.1
                    }
                }
            },
            "beta_1": {
                "hyper_opt": {
                    "suggest": "loguniform",
                    "params": {
                        "name": "beta_1",
                        "low": 0.00001,
                        "high": 0.1
                    }
                }
            }
        }
    }


def test_processor(base_processor: SearchTemplateProcessor, template: Dict[str, Any]):
    resolved_config = base_processor.modify(template)

    assert type(resolved_config["model"]["layer_size"]) == int
    assert type(resolved_config["trainer"]["learning_rate"]) == float
    assert type(resolved_config["trainer"]["beta_1"]) == float
