import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.model_selection import train_test_split  # noqa: E402

from datasets import ALL  # noqa: E402
from models import build, candidates  # noqa: E402


def test_all_datasets_load_without_leaking_target():
    for loader in ALL.values():
        task = loader()
        assert len(task.X) == len(task.y) > 100
        assert task.y.name not in task.X.columns
        assert set(task.numeric + task.categorical) == set(task.X.columns)


def test_every_model_fits_and_predicts():
    for loader in ALL.values():
        task = loader()
        X_tr, X_te, y_tr, _ = train_test_split(task.X, task.y, test_size=0.3, random_state=0)
        for model, _ in candidates(task.kind).values():
            pipe = build(model, task.numeric, task.categorical).fit(X_tr.head(300), y_tr.head(300))
            assert len(pipe.predict(X_te.head(20))) == 20
