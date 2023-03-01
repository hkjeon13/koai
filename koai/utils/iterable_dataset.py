from typing import Optional, Any, List
from datasets import interleave_datasets
from datasets.combine import concatenate_datasets
from torch.utils.data import IterableDataset


def nrows_from_info(dataset: Any, split_name: str = 'train'):
    num_rows = -1
    if hasattr(dataset, "info") and hasattr(dataset.info, "splits"):
        _split = dataset.info.splits
        if _split is not None:
            sp_info = _split.get(split_name)
            if sp_info and hasattr(sp_info, "num_examples"):
                num_rows = sp_info.num_examples
    return num_rows


class IterableDatasetWrapper(IterableDataset):
    def __init__(self,
                 datasets: List[IterableDataset],
                 split_names: List[str] = None,
                 length: Optional[int] = None,
                 max_rows: int = 1000000,
                 data_format: str = "torch",
                 merge_method: str = "concatenate",
                 interleave_probs: List[str] = None,
                 each_data_shuffle: bool = False) -> None:

        super(IterableDatasetWrapper, self).__init__()
        split_names = ['train'] * len(datasets) if split_names is None else split_names

        assert len(datasets) == len(split_names)

        if isinstance(interleave_probs, list):
            assert len(interleave_probs) == len(datasets)

        _datasets, _lengths = [], []
        for dataset, split_name in zip(datasets, split_names):
            _length = nrows_from_info(dataset, split_name)
            _length = max_rows if _length < 0 else _length
            dataset = dataset.shuffle() if each_data_shuffle else dataset
            _datasets.append(dataset.with_format(data_format))
            _lengths.append(_length)

        if merge_method == "concatenate":
            self.dataset = concatenate_datasets(_datasets)

        elif merge_method == "interleave":
            self.dataset = interleave_datasets(_datasets, probabilities=interleave_probs)

        self.length = sum(_lengths) if length is None else length

    def __iter__(self):
        _gen = iter(self.dataset)
        for i in range(self.length):
            try:
                yield next(_gen)
            except StopIteration:
                break

    def __len__(self) -> int:
        return self.length

    def take(self, n: int):
        _gen = iter(self.dataset)
        for i in range(min(self.length, n)):
            try:
                yield next(_gen)
            except StopIteration:
                break

    def select(self, ids: List[int]):
        _gen = iter(self.dataset)
        _ids = set(ids)
        for i in range(self.length):
            if i in _ids:
                continue
            try:
                yield next(_gen)
            except StopIteration:
                break


if __name__ == "__main__":
    from datasets import load_dataset

    dataset1 = load_dataset("psyche/kowiki", streaming=True)
    dataset2 = load_dataset("psyche/common_crawl", "1", streaming=True)
    dataset = IterableDatasetWrapper([dataset1['train'], dataset2['train']], split_names=['train', 'train'],
                                     merge_method='concatenate')
    print(len(dataset))
