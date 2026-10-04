"""Run-local, byte-bounded cache of exact CPU preprocessing results."""
from collections import OrderedDict
import warnings

import torch


class PreprocessingCache:
    """Budget covers resident tensor payloads, not batches/decoder/allocator overhead.

    No disk persistence or device allocations. Callers provide a namespace tied to
    the validated session, accepted samples, alignment and preprocessing. Returned
    clones keep later tensor mutation from changing subsequent examples.
    """

    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('cache max_bytes must be a nonnegative integer')
        self.max_bytes = max_bytes
        self.resident_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self._entries = OrderedDict()

    def _allocation_failure(self):
        self._entries.clear()
        self.resident_bytes = 0
        self.max_bytes = 0
        warnings.warn('preprocessing cache allocation failed; continuing uncached', RuntimeWarning)

    @staticmethod
    def _is_oom(error):
        # CPU allocators in some supported PyTorch releases raise RuntimeError.
        return (isinstance(error, (MemoryError, torch.OutOfMemoryError))
                or "can't allocate memory" in str(error).lower()
                or 'not enough memory' in str(error).lower())

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        try:
            result = tuple(value.clone() for value in entry[0])
        except (MemoryError, RuntimeError) as error:
            if not self._is_oom(error):
                raise
            self._allocation_failure()
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return result

    def put(self, key, values):
        size = sum(value.numel() * value.element_size() for value in values)
        if size > self.max_bytes or self.max_bytes == 0:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.resident_bytes -= previous[1]
        while self.resident_bytes + size > self.max_bytes:
            _, (_, removed) = self._entries.popitem(last=False)
            self.resident_bytes -= removed
            self.evictions += 1
        try:
            stored = tuple(value.clone() for value in values)
        except (MemoryError, RuntimeError) as error:
            if not self._is_oom(error):
                raise
            self._allocation_failure()
            return
        self._entries[key] = stored, size
        self.resident_bytes += size
