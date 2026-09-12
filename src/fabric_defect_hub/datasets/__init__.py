"""DatasetAdapter implementations. Each concrete dataset module should
subclass `DatasetAdapter` (see `base.py`) and register itself with
`@fabric_defect_hub.core.registry.register_dataset("<name>")`.

Importing this package registers all bundled datasets.
"""

from fabric_defect_hub.datasets.anomalib_folder import anomalib_folder_staging_dir
from fabric_defect_hub.datasets.fabric_defects import FabricDefectsDataset
from fabric_defect_hub.datasets.fabric_train import FabricTrainDataset
from fabric_defect_hub.datasets.mvtec_ad import MVTecADDataset
from fabric_defect_hub.datasets.mvtec_loco import MVTecLOCODataset
from fabric_defect_hub.datasets.raw_fabric import RawFabricDataset
from fabric_defect_hub.datasets.tianchi import TianchiDataset
from fabric_defect_hub.datasets.tilda import TILDA400Dataset
from fabric_defect_hub.datasets.visa import VisADataset
from fabric_defect_hub.datasets.yolo_bbox import build_class_map, yolo_staging_dir
from fabric_defect_hub.datasets.zju_leaper import ZJULeaperDataset

__all__ = [
    "ZJULeaperDataset",
    "RawFabricDataset",
    "MVTecADDataset",
    "MVTecLOCODataset",
    "TILDA400Dataset",
    "FabricDefectsDataset",
    "VisADataset",
    "FabricTrainDataset",
    "TianchiDataset",
    "yolo_staging_dir",
    "build_class_map",
    "anomalib_folder_staging_dir",
]
