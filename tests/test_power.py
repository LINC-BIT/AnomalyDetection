import pytest

from fabric_defect_hub.profiling.power import (
    PowerCapability,
    PowerMonitor,
    PowerSample,
    _parse_tegrastats_watts,
    _parse_powermetrics_watts,
    assess_power_capability,
    summarize_power_samples,
)


def test_assess_jetson_prefers_board_power():
    capability = assess_power_capability(
        "cuda:0", system="Linux", is_jetson=True, is_raspberry_pi=False,
        command_exists=lambda command: command == "tegrastats",
    )
    assert capability.available is True
    assert capability.source == "tegrastats"
    assert capability.scope == "board_input"


def test_assess_macos_reports_powermetrics_scope():
    capability = assess_power_capability(
        system="Darwin", is_jetson=False, is_raspberry_pi=False,
        command_exists=lambda command: command == "powermetrics",
    )
    assert capability == PowerCapability("macos", "powermetrics", "package", True)


def test_assess_raspberry_pi_requires_sensor():
    capability = assess_power_capability(
        system="Linux", is_jetson=False, is_raspberry_pi=True, sensor_paths=[],
    )
    assert capability.available is False
    assert "INA219" in capability.setup_hint


def test_parse_powermetrics_millwatts_and_watts():
    assert _parse_powermetrics_watts("System Power: 5234 mW") == pytest.approx(5.234)
    assert _parse_powermetrics_watts("Combined Power: 12.5 W") == 12.5
    assert _parse_powermetrics_watts("no power here") is None


def test_parse_tegrastats_uses_latest_board_input_reading():
    output = "RAM 1/2MB VDD_IN 4300mW\nRAM 1/2MB VDD_IN 5100mW"
    assert _parse_tegrastats_watts(output) == 5.1


def test_power_summary_is_time_weighted():
    samples = [PowerSample(10.0, 2.0), PowerSample(12.0, 6.0)]
    average, peak, energy = summarize_power_samples(samples, duration_s=2.0)
    assert average == 4.0
    assert peak == 6.0
    assert energy == 8.0


def test_required_unavailable_power_fails_explicitly():
    capability = PowerCapability("linux", "none", "unknown", False, "no sensor")
    monitor = PowerMonitor(capability, None, required=True)
    with pytest.raises(RuntimeError, match="required but unavailable"):
        monitor.start()


def test_auto_unavailable_power_returns_report_without_metrics():
    capability = PowerCapability("linux", "none", "unknown", False, "no sensor")
    monitor = PowerMonitor(capability, None)
    monitor.start()
    report = monitor.stop()
    assert report.status == "unavailable"
    assert report.metrics() == {}
