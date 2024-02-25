from typing import Tuple

from type_extensions import DataPointTest, DataPointTrain, DataPointType


def split_train_test_dataset(
    data: DataPointType, split_size: float = 0.1
) -> Tuple[DataPointTrain, DataPointTest]:
    data = data.train_test_split(test_size=split_size)
    train_data = data["train"]
    test_data = data["test"]
    return train_data, test_data
