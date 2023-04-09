import pytest
from pathlib import Path
from labeler.trainer import ClassifierTrainer, TrainingConfig, Metrics


@pytest.fixture(scope="module")
def training_config():
    return TrainingConfig(
        labels=Path("./tests/fixtures/doctype/labels.csv"), batch_size=2, learning_rate=2e-5, num_epochs=10
    )


@pytest.fixture(scope="module")
def trainer(training_config: TrainingConfig):
    return ClassifierTrainer(log_files_directory=Path("./tests/fixtures/doctype"), training_config=training_config)


@pytest.mark.integration
def test_trainer(trainer: ClassifierTrainer):
    metrics = trainer.train()
    assert isinstance(metrics, Metrics)
    assert metrics.average_loss > 0.0
    assert metrics.average_loss < 0.15
    assert metrics.fpr == 0.0
    assert metrics.fnr == 0.0
    assert metrics.accuracy == 1.0
