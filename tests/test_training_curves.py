from fabric_defect_hub.reporting.training_curves import load_training_curve, render_training_curve_svg


def test_training_curve_renders_svg(tmp_path):
    history = tmp_path / "history.csv"
    history.write_text("epoch,train_loss,val_map\n0,1.0,0.2\n1,0.5,0.4\n")

    curve = load_training_curve(history)
    output = render_training_curve_svg(curve, tmp_path / "curve.svg")

    assert curve.x_name == "epoch"
    assert "train_loss" in curve.series
    assert "<polyline" in output.read_text()
