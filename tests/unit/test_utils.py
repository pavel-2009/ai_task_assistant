import importlib
from types import SimpleNamespace

import torch

from app.utils.torch_basic import grad_study


class _DummyModel:
    def eval(self):
        return self

    def __call__(self, _):
        out = torch.zeros((1, 3), dtype=torch.float32)
        out[0, 1] = 1.0
        return out


def test_cv_model_without_real_model(monkeypatch):
    import torchvision.models as models

    monkeypatch.setattr(models, 'resnet18', lambda weights=None: _DummyModel())
    monkeypatch.setattr(
        models,
        'ResNet18_Weights',
        SimpleNamespace(IMAGENET1K_V1=SimpleNamespace(meta={'categories': ['cat', 'dog', 'house']})),
    )

    cv_model = importlib.reload(importlib.import_module('app.utils.cv_model'))
    assert hasattr(cv_model, 'predict_image_class')


def test_torch_basic_gradients(capsys):
    grad_study()
    out = capsys.readouterr().out
    assert 'x.grad:' in out
