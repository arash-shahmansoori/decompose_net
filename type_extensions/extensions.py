from typing import List, TypeVar

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

T = TypeVar("T")

DataPointType = DatasetDict | Dataset | IterableDatasetDict | IterableDataset
DataPointTrain = Dataset | List | T
DataPointTest = Dataset | List | T
