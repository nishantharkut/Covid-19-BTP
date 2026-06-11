import numpy as np

from covid_audio_btp.preprocess import crop_or_pad_audio


def test_crop_or_pad_audio_pads_short_signal():
    y = np.ones(4, dtype=np.float32)
    out = crop_or_pad_audio(y, target_samples=8)

    assert out.shape == (8,)
    assert out[:2].tolist() == [0.0, 0.0]
    assert out[2:6].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert out[6:].tolist() == [0.0, 0.0]


def test_crop_or_pad_audio_center_crops_long_signal():
    y = np.arange(10, dtype=np.float32)
    out = crop_or_pad_audio(y, target_samples=4)

    assert out.tolist() == [3.0, 4.0, 5.0, 6.0]

