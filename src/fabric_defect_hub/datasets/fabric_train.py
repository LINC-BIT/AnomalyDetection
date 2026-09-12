"""`fabric-train` -- a *composite* dataset that unions the in-domain fabric
datasets into a single training corpus.

This is the mechanism behind the project's training policy (see
`training.ANOMALY_TRAINABLE_DATASETS`): anomaly models should train on more
than one fabric dataset, but not on arbitrary/cross-domain ones. Rather than
change every backend's single-`dataset` config schema, `fabric-train` is
itself a registered `DatasetAdapter` whose `load_samples()` concatenates its
members' samples -- so `data.dataset: fabric-train` flows through the
existing one-dataset pipeline unchanged.

Members (each a fabric, in-domain anomaly source) are every dataset declared
with the "fabric_train_member" role in `core.dataset_capabilities` (ZJU-Leaper,
RAW_FABRID, TILDA-400, Fabric Defects Dataset, Tianchi as of this writing) --
adding a new fabric dataset's capability declaration is enough to fold it
into this union, no edit needed here. Each member is resolved through the
same registry the loader uses, with a root of `<root>/<member subdir>`
(default `root` is the project's `data/` directory).

Selection semantics match the other adapters, applied to the *union*:
`num_samples` is the global total across all members (None = everything);
`use_defect`/`defect_ratio` control the normal/defect mix globally;
`split`/`task`/`seed` are forwarded to every member. Member-specific knobs
(ZJU-Leaper's `pattern`, ...) are intentionally not exposed here -- the whole
point is to draw from every texture/source at once.
"""

from __future__ import annotations

import random
from pathlib import Path

from fabric_defect_hub.core.dataset_capabilities import all_capabilities
from fabric_defect_hub.core.registry import get_dataset_cls, register_dataset
from fabric_defect_hub.core.types import Sample, Task
from fabric_defect_hub.datasets.base import DatasetAdapter


def _members() -> tuple[tuple[str, str], ...]:
    """(registry name, subdirectory under `root`) for every dataset declared
    with the "fabric_train_member" role -- each member's default root is
    `data/<subdir>` (see `core.dataset_capabilities`), so the subdir is just
    that default root with the `data/` prefix stripped."""

    members = [
        (name, Path(caps.default_root).name)
        for name, caps in all_capabilities().items()
        if "fabric_train_member" in caps.roles and caps.default_root
    ]
    return tuple(sorted(members))


_MEMBERS: tuple[tuple[str, str], ...] = _members()


@register_dataset("fabric-train")
class FabricTrainDataset(DatasetAdapter):
    """Union of the in-domain fabric datasets, for one-class training.

    root: the base directory holding each member (default "data"). Member i
        is instantiated with root `<root>/<member subdir>`.
    split/num_samples/use_defect/defect_ratio/task/seed: as in every other
        anomaly adapter, applied to the concatenated union (see module
        docstring).
    """

    name = "fabric-train"

    def __init__(
        self,
        root: str = "datasets/textile",
        split: str = "train",
        num_samples: int | None = None,
        use_defect: bool = False,
        defect_ratio: float = 0.5,
        task: Task = "anomaly",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(root=root, split=split, **kwargs)
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if not 0.0 <= defect_ratio <= 1.0:
            raise ValueError(f"defect_ratio must be in [0, 1], got {defect_ratio}")

        self.root_path = Path(root)
        self.num_samples = num_samples
        self.use_defect = use_defect
        self.defect_ratio = defect_ratio
        self.task = task
        self.seed = seed

    def _member_samples(self) -> tuple[list[Sample], list[Sample]]:
        """Load every member and split its samples into (normal, defect),
        prefixing ids with the member name to keep them globally unique.
        Members whose directory is absent are skipped (a partial local
        staging shouldn't hard-fail the whole union).
        """

        normal: list[Sample] = []
        defect: list[Sample] = []
        for member_name, subdir in _MEMBERS:
            member_root = self.root_path / subdir
            if not member_root.exists():
                continue
            cls = get_dataset_cls(member_name)
            # num_samples=None here: subsampling to the global budget happens
            # once over the union below, not per-member.
            member = cls(
                root=str(member_root),
                split=self.split,
                num_samples=None,
                use_defect=self.use_defect,
                defect_ratio=self.defect_ratio,
                task=self.task,
                seed=self.seed,
            )
            for sample in member.load_samples():
                tagged = Sample(
                    id=f"{member_name}/{sample.id}",
                    image_path=sample.image_path,
                    task=sample.task,
                    annotations=sample.annotations,
                    metadata={**sample.metadata, "source_dataset": member_name},
                )
                (defect if sample.annotations.is_anomalous else normal).append(tagged)
        return normal, defect

    def load_samples(self) -> list[Sample]:
        normal, defect = self._member_samples()

        rng = random.Random(self.seed)
        rng.shuffle(normal)
        rng.shuffle(defect)

        if self.num_samples is None:
            return normal + defect
        if not defect:
            return normal[: self.num_samples]

        n_defect = min(round(self.num_samples * self.defect_ratio), len(defect))
        n_normal = min(self.num_samples - n_defect, len(normal))
        return normal[:n_normal] + defect[:n_defect]
