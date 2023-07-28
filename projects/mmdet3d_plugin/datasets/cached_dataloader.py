from typing import Optional
import torch

class CachedDataLoader:
    """Caches a real dataloader's first outputs.
    Useful for measuring model-only throughput.
    """
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
        num_batches: int = 4,
    ):
        """Constructor.
        Args:
            dataloader: The original torch dataloader whose first outputs should be cached.
            device: The device to which the dataloader outputs will be copied to. Default to None.
            num_batches: Number of batches to cache. This increases the diversity of
                training samples.
        """
        self._dataloader = dataloader
        self._num_batches = min(num_batches, len(dataloader))
        self._device = device
        self._outputs = None
        self._curr_index = 0
    def __len__(self) -> int:
        """Length."""
        return len(self._dataloader)
    def __iter__(self):
        """Iterate through the dataloader."""
        for _ in range(len(self)):
            if self._outputs is None:
                self._outputs = []
                iterator = iter(self._dataloader)
                for _ in range(self._num_batches):
                    sample = next(iterator, None)
                    if sample is None:
                        # This should rarely happen since we set self._num_batches to be
                        # at most len(dataloader). This just handles the case where dataloader
                        # is an empty iterator.
                        return
                    self._outputs.append(sample)
            yield self._outputs[self._curr_index]
            self._curr_index = (self._curr_index + 1) % self._num_batches
    def __getattr__(self, key):
        """Get attribute.
        This allows users of this class to access attributes as if it were a torch dataloader.
        """
        # Only invoked if the attribute wasn't found the usual ways.
        return getattr(self._dataloader, key)

