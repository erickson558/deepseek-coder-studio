import random
from typing import Sequence

from app.models.dataset import DatasetExample


def split_dataset(
    examples: Sequence[DatasetExample],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[DatasetExample], list[DatasetExample], list[DatasetExample]]:
    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)

    train_end = int(len(items) * train_ratio)
    validation_end = train_end + int(len(items) * validation_ratio)
    train_split = items[:train_end]
    validation_split = items[train_end:validation_end]
    test_split = items[validation_end:]
    return train_split, validation_split, test_split
