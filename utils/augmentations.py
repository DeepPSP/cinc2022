"""
Audio data augmentation using `torch_audiomentations`
"""

from typing import Sequence, NoReturn

import torch_audiomentations as TA
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform


class Augmenter(TA.SomeOf):
    """

    Audio data augmenters

    """

    def __init__(
        self,
        transforms: Sequence[BaseWaveformTransform],
        p: float = 1.0,
    ) -> NoReturn:
        """

        """
        super().__init__((1, None), transforms, p=p)
