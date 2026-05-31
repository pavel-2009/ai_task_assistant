from app.ml.common.config import MLConfig


def test_default_paths():
    cfg = MLConfig()
    assert str(cfg.data_dir).endswith('app/avatars')
    assert str(cfg.output_dir).endswith('app/checkpoints')


def test_training_params():
    cfg = MLConfig()
    assert cfg.batch_size == 16
    assert cfg.learning_rate == 0.001
    assert cfg.num_epochs == 20


def test_normalization_mean_std():
    cfg = MLConfig()
    assert cfg.mean == [0.485, 0.456, 0.406]
    assert cfg.std == [0.229, 0.224, 0.225]


def test_image_size():
    cfg = MLConfig()
    assert cfg.img_size == 224
