from fabric_defect_hub.core.checkpoint import inspect_checkpoint


def test_checkpoint_diagnostic_reports_a_missing_file_without_loading_it(tmp_path):
    diagnostic = inspect_checkpoint(tmp_path / "missing.ckpt")

    assert diagnostic.exists is False
    assert diagnostic.sha256 is None
    assert diagnostic.unsafe_globals == ()
