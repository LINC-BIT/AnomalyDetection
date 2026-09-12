"""Fast, framework-free tests for the MoECLIP backend's declarative layer:
`MoECLIPConfig` parsing/validation, the prompt-name resolution the model's
text branch depends on, and the `Sample` -> batch bridge's sample
filtering. Nothing here builds a model or imports torch — the vendored
checkout and its CLIP backbone checkpoint aren't available in CI (see
`components/README.md`), so the end-to-end path is exercised by hand
against a real checkout instead.
"""

import pytest

from fabric_defect_hub.core.types import Annotations, Sample
from fabric_defect_hub.models.moeclip import presets
from fabric_defect_hub.models.moeclip.config import MoECLIPConfig

MINIMAL = {
    "model": {"name": "ViT-L-14-336"},
    "data": {
        # Zero-shot protocol: train on an auxiliary cross-domain corpus,
        # evaluate on fabric the model has never seen.
        "dataset": "visa",
        "dataset_root": "data/VisA",
        "test_dataset": "raw-fabric",
        "test_dataset_root": "data/RAW_FABRID",
        "train_selection": {"split": "train", "use_defect": True, "task": "segmentation"},
        "test_selection": {"split": "test", "use_defect": True},
    },
}


def _sample(sample_id: str, *, defect: bool, mask: str | None = None, category=None) -> Sample:
    return Sample(
        id=sample_id,
        image_path=f"/tmp/{sample_id}.png",
        task="anomaly",
        annotations=Annotations(is_anomalous=defect, anomaly_mask=mask),
        metadata={"category": category} if category else {},
    )


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #
def test_from_dict_applies_upstream_defaults():
    config = MoECLIPConfig.from_dict(MINIMAL)
    assert config.model.img_size == 518
    assert config.model.moe_layers == [5, 11, 17, 23]
    assert config.train.epochs == 20
    assert config.train.batch_size == 2
    assert config.checkpoint.registry_dir == "artifacts/models"


def test_resolved_train_kwargs_merges_model_architecture():
    config = MoECLIPConfig.from_dict({**MINIMAL, "model": {"name": "ViT-L-14-336", "moe_top_k": 1}})
    kwargs = config.resolved_train_kwargs()
    # Architecture knobs travel with the training kwargs so the adapter can
    # record them on the artifact.
    assert kwargs["moe_top_k"] == 1
    assert kwargs["img_size"] == 518
    assert kwargs["epochs"] == 20


def test_unknown_backbone_is_rejected():
    with pytest.raises(KeyError, match="unknown MoECLIP backbone"):
        MoECLIPConfig.from_dict({**MINIMAL, "model": {"name": "ViT-B-16"}})


def test_unknown_section_key_is_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        MoECLIPConfig.from_dict({**MINIMAL, "train": {"epochs": 1, "nope": 2}})


def test_dataset_is_required():
    with pytest.raises(ValueError, match="'dataset' is required"):
        MoECLIPConfig.from_dict({**MINIMAL, "data": {"dataset_root": "data/VisA"}})


def test_dataset_root_is_required():
    with pytest.raises(ValueError, match="requires 'dataset_root'"):
        MoECLIPConfig.from_dict({**MINIMAL, "data": {"dataset": "visa"}})


def test_test_dataset_requires_its_own_root():
    data = {k: v for k, v in MINIMAL["data"].items() if k != "test_dataset_root"}
    with pytest.raises(ValueError, match="test_dataset='raw-fabric' requires"):
        MoECLIPConfig.from_dict({**MINIMAL, "data": data})


def test_eval_dataset_is_the_zero_shot_target():
    config = MoECLIPConfig.from_dict(MINIMAL)
    assert config.data.eval_dataset() == ("raw-fabric", "data/RAW_FABRID")


def test_eval_dataset_falls_back_to_the_training_corpus():
    data = {k: v for k, v in MINIMAL["data"].items() if not k.startswith("test_dataset")}
    config = MoECLIPConfig.from_dict({**MINIMAL, "data": data})
    assert config.data.eval_dataset() == ("visa", "data/VisA")


def test_invalid_seg_proj_sharing_strategy_is_rejected():
    with pytest.raises(ValueError, match="seg_proj_sharing_strategy"):
        MoECLIPConfig.from_dict(
            {**MINIMAL, "model": {"name": "ViT-L-14-336", "seg_proj_sharing_strategy": "both"}}
        )


def test_shipped_example_config_parses():
    from fabric_defect_hub.training import (
        ANOMALY_TRAINABLE_DATASETS,
        ZERO_SHOT_TRAINABLE_DATASETS,
        apply_default_dataset_root,
        load_raw_config,
    )

    raw = apply_default_dataset_root(load_raw_config("configs/models/moeclip_example.yaml"))
    config = MoECLIPConfig.from_dict(raw)
    # The train split must carry defects: MoECLIP learns from labelled
    # anomalies, unlike the one-class backends.
    assert config.data.train_selection["use_defect"] is True
    assert config.val.enabled is True
    # ...and it must be the zero-shot protocol: auxiliary corpus in,
    # fabric out.
    assert config.data.dataset in ZERO_SHOT_TRAINABLE_DATASETS
    assert config.data.eval_dataset()[0] in ANOMALY_TRAINABLE_DATASETS


# --------------------------------------------------------------------- #
# Prompt resolution -- the one seam between this project's data contracts
# and MoECLIP's text branch (see models/moeclip/presets.py).
# --------------------------------------------------------------------- #
def test_class_name_falls_back_to_fabric():
    assert presets.class_name_for(_sample("a", defect=False)) == "fabric"


def test_class_name_uses_dataset_category_when_present():
    assert presets.class_name_for(_sample("a", defect=False, category="metal_nut")) == "metal_nut"


def test_class_name_uses_zju_leaper_pattern_name():
    sample = _sample("a", defect=False)
    sample.metadata = {"fabric_type": "dot pattern"}
    assert presets.class_name_for(sample) == "dot pattern"


def test_defect_type_is_never_used_as_a_prompt_class():
    # Prompting with the defect type would leak the label being predicted.
    sample = _sample("a", defect=True, mask="/tmp/m.png")
    sample.metadata = {"defect_type": "hole"}
    assert presets.class_name_for(sample) == "fabric"
    assert "defect_type" not in presets.CLASS_METADATA_KEYS


def test_prompt_class_pins_every_sample():
    sample = _sample("a", defect=False, category="metal_nut")
    assert presets.class_name_for(sample, forced="fabric") == "fabric"


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [("fabric", "fabric texture"), ("metal_nut", "metal nut"), ("pipe_fryum", "pipe fryum")],
)
def test_real_name_for(class_name, expected):
    assert presets.real_name_for(class_name) == expected


def test_real_name_overrides_win():
    assert presets.real_name_for("fabric", {"fabric": "plain woven cotton"}) == "plain woven cotton"


def test_prompt_policy_is_not_part_of_the_architecture():
    # Prompts change no weight shape, so they must not travel in the
    # artifact metadata that rebuilds the model -- MoECLIP trains on
    # objects and is run on fabric, with different prompts by design.
    config = MoECLIPConfig.from_dict(
        {**MINIMAL, "model": {"name": "ViT-L-14-336", "prompt_class": "fabric",
                              "prompts": {"fabric": "plain woven cotton fabric"}}}
    )
    assert "prompt_class" not in config.model.arch_kwargs()
    assert config.model.adapter_kwargs()["prompt_class"] == "fabric"


# --------------------------------------------------------------------- #
# Train-split filtering
# --------------------------------------------------------------------- #
def _select(samples):
    from fabric_defect_hub.models.moeclip.adapter import MoECLIPAdapter

    return MoECLIPAdapter._select_train_samples(object.__new__(MoECLIPAdapter), samples)


def test_defective_samples_without_masks_are_dropped():
    samples = [
        _sample("n1", defect=False),
        _sample("d1", defect=True, mask="/tmp/d1_mask.png"),
        _sample("d2", defect=True),  # no pixel ground truth
    ]
    kept, stats = _select(samples)
    assert [s.id for s in kept] == ["n1", "d1"]
    assert stats == {"train_samples": 2, "train_defective": 1, "dropped_unmasked_defects": 1}


def test_normal_only_train_split_is_rejected():
    with pytest.raises(ValueError, match="needs defective samples with pixel masks"):
        _select([_sample("n1", defect=False), _sample("n2", defect=False)])


def test_all_defects_unmasked_is_rejected():
    with pytest.raises(ValueError, match="no usable MoECLIP training samples|needs defective"):
        _select([_sample("d1", defect=True)])
