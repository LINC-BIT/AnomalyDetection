"""`DatasetAdapter` for TILDA-400, a fabric-texture defect dataset laid out
as flat class folders: `good/` (normal) plus `hole/`, `oil spot/`,
`thread error/`, `objects/` (defect types). Image-level labels only — there
are no pixel masks — and no author-provided train/test split, so this reuses
`FlatFolderAnomalyDataset`'s synthesized, leak-free split (see its docstring).

In-domain fabric, so it is a *training-eligible* source (see
`training.ANOMALY_TRAINABLE_DATASETS` and the `fabric-train` composite), and
also usable on its own for image-level anomaly evaluation.
"""

from __future__ import annotations

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.datasets.flat_folder import FlatFolderAnomalyDataset


@register_dataset("tilda-400")
class TILDA400Dataset(FlatFolderAnomalyDataset):
    name = "tilda-400"
    NORMAL_DIRNAME = "good"
