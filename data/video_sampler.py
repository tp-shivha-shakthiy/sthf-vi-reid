import random


class VideoSampler:
    """Identity-balanced video sampler.

    Yields batches of indices where each batch contains clips from
    ``num_ids`` distinct identities, with ``clips_per_id`` clips per
    identity — producing a total batch size of ``num_ids * clips_per_id``.

    Args:
        dataset: A dataset instance that supports ``dataset.samples``
            where each sample dict has a ``"pid"`` key.
        num_ids: Number of distinct identities per batch.
        clips_per_id: Number of clips sampled per identity.
        shuffle: Whether to shuffle identities and clips each epoch.
    """

    def __init__(self, dataset, num_ids=4, clips_per_id=4, shuffle=True):
        self.dataset = dataset
        self.num_ids = num_ids
        self.clips_per_id = clips_per_id
        self.batch_size = num_ids * clips_per_id
        self.shuffle = shuffle

        # Build PID -> list of sample indices
        self.pid_to_indices = {}
        for idx, sample in enumerate(dataset.samples):
            pid = sample["pid"]
            self.pid_to_indices.setdefault(pid, []).append(idx)

        self.pids = list(self.pid_to_indices.keys())
        self._epoch = 0

    def __len__(self):
        """Number of batches per epoch (approximate)."""
        total = sum(len(v) for v in self.pid_to_indices.values())
        return max(1, total // self.batch_size)

    def __iter__(self):
        """Yield batches of sample indices."""
        pid_pool = list(self.pids)
        if self.shuffle:
            random.shuffle(pid_pool)

        # Build per-PID index pools
        pid_indices = {
            pid: list(indices)
            for pid, indices in self.pid_to_indices.items()
        }
        if self.shuffle:
            for v in pid_indices.values():
                random.shuffle(v)

        batch = []
        pid_ptr = 0

        while pid_ptr < len(pid_pool):
            # Pick num_ids consecutive PIDs (wrap around not needed since
            # we reshuffle each epoch)
            if pid_ptr + self.num_ids > len(pid_pool):
                break

            for _ in range(self.num_ids):
                pid = pid_pool[pid_ptr]
                pool = pid_indices[pid]

                # Sample clips_per_id indices for this PID
                chosen = pool[:self.clips_per_id]
                if len(chosen) < self.clips_per_id:
                    # Pad with random repeats if not enough samples
                    chosen = chosen + random.choices(pool,
                                                     k=self.clips_per_id - len(chosen))
                batch.extend(chosen)

                # Remove used indices (to avoid re-sampling the same clip)
                pid_indices[pid] = pool[self.clips_per_id:]
                if not pid_indices[pid]:
                    pid_indices[pid] = list(self.pid_to_indices[pid])
                    if self.shuffle:
                        random.shuffle(pid_indices[pid])

                pid_ptr += 1

            yield batch
            batch = []

        self._epoch += 1
