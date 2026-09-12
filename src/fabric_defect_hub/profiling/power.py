"""Cross-platform power capability assessment and sampled power reporting.

The platform decision is explicit: a missing sensor is reported as
``unavailable`` rather than a misleading 0 W. NVIDIA servers use NVML GPU
power, Jetson prefers ``tegrastats`` board-input power, macOS uses the SMC
sampler exposed by ``powermetrics``, and Raspberry Pi uses a kernel-exported
power sensor such as INA219/INA226.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import time
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class PowerCapability:
    platform: str
    source: str
    scope: str
    available: bool
    reason: str | None = None
    setup_hint: str | None = None


@dataclass(frozen=True)
class PowerSample:
    timestamp_s: float
    watts: float


@dataclass
class PowerReport:
    capability: PowerCapability
    status: str
    sample_count: int
    duration_s: float
    average_watts: float | None = None
    peak_watts: float | None = None
    energy_joules: float | None = None
    reason: str | None = None

    def metrics(self) -> dict[str, float]:
        if self.status != "measured" or self.average_watts is None:
            return {}
        return {
            "power_w_mean": self.average_watts,
            "power_w_peak": self.peak_watts or self.average_watts,
            "energy_j": self.energy_joules or 0.0,
            "power_sample_count": float(self.sample_count),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def save_power_report(report: PowerReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def assess_power_capability(
    device: str = "auto",
    *,
    system: str | None = None,
    is_jetson: bool | None = None,
    is_raspberry_pi: bool | None = None,
    command_exists: Callable[[str], bool] | None = None,
    sensor_paths: list[Path] | None = None,
) -> PowerCapability:
    """Report the usable power source and measurement scope for this host.

    ``device`` selects an NVIDIA GPU when it begins with ``cuda`` or equals
    ``gpu``. On Jetson we deliberately prefer board-input power from
    ``tegrastats`` over GPU-only NVML power, because edge deployment needs
    the power actually drawn by the device.
    """

    system = system or platform.system()
    command_exists = command_exists or (lambda command: shutil.which(command) is not None)
    is_jetson = _is_jetson() if is_jetson is None else is_jetson
    is_raspberry_pi = _is_raspberry_pi() if is_raspberry_pi is None else is_raspberry_pi

    if is_jetson:
        if command_exists("tegrastats"):
            return PowerCapability("jetson", "tegrastats", "board_input", True)
        return PowerCapability(
            "jetson", "tegrastats", "board_input", False,
            "tegrastats is not available.", "Install the NVIDIA JetPack tegrastats utility."
        )

    if is_raspberry_pi:
        paths = sensor_paths if sensor_paths is not None else _raspberry_power_paths()
        if paths:
            return PowerCapability("raspberry_pi", "sysfs", "sensor_rail", True)
        return PowerCapability(
            "raspberry_pi", "sysfs", "sensor_rail", False,
            "No kernel-exported power sensor was found.",
            "Attach INA219/INA226 or another supported power sensor and expose its power*_input sysfs file."
        )

    if system == "Darwin":
        if command_exists("powermetrics"):
            return PowerCapability("macos", "powermetrics", "package", True)
        return PowerCapability(
            "macos", "powermetrics", "package", False,
            "powermetrics is not available.", "Use macOS with the built-in powermetrics utility."
        )

    if device.startswith("cuda") or device == "gpu":
        try:
            import pynvml  # noqa: F401
        except ImportError:
            return PowerCapability(
                "nvidia", "nvml", "gpu", False,
                "The nvidia-ml-py package is not installed.",
                "Install fabric-defect-hub[profiling-power-nvidia] on the NVIDIA host."
            )
        return PowerCapability("nvidia", "nvml", "gpu", True)

    return PowerCapability(
        system.lower(), "none", "unknown", False,
        "No supported power sensor is configured for this platform.",
        "Use NVIDIA CUDA, macOS powermetrics, or a Raspberry Pi external power sensor."
    )


class PowerMonitor:
    """Collect periodic instantaneous power samples during measured inference."""

    def __init__(
        self,
        capability: PowerCapability,
        reader: Callable[[], float] | None,
        sample_interval_ms: int = 100,
        required: bool = False,
    ):
        self.capability = capability
        self._reader = reader
        self._interval_s = max(sample_interval_ms, 1) / 1000.0
        self._required = required
        self._samples: list[PowerSample] = []
        self._started_at: float | None = None
        self._last_sample_at: float | None = None
        self._failure: str | None = None

    @classmethod
    def from_profile_config(cls, config) -> "PowerMonitor":
        mode = getattr(config, "power_mode", "auto")
        capability = assess_power_capability(
            config.device,
            sensor_paths=[Path(config.power_sensor_path)] if getattr(config, "power_sensor_path", None) else None,
        )
        if mode == "disabled":
            capability = PowerCapability(capability.platform, capability.source, capability.scope, False, "Disabled by profile config.")
            return cls(capability, None, required=False)
        if not capability.available:
            return cls(capability, None, required=mode == "required")
        return cls(
            capability,
            _reader_for(capability, config.device, getattr(config, "power_sensor_path", None)),
            sample_interval_ms=getattr(config, "power_sample_interval_ms", 100),
            required=mode == "required",
        )

    def start(self) -> None:
        if not self.capability.available or self._reader is None:
            if self._required:
                raise RuntimeError(_power_unavailable_message(self.capability))
            return
        # Establish a baseline before starting the measured window. This
        # matters for command-based samplers (notably macOS powermetrics):
        # their own collection time must not inflate model energy.
        self._started_at = time.monotonic()
        self.sample(force=True)
        self._started_at = time.monotonic()
        if self._samples:
            self._samples[-1] = PowerSample(self._started_at, self._samples[-1].watts)
            self._last_sample_at = self._started_at

    def sample(self, force: bool = False, timestamp_s: float | None = None) -> None:
        if self._started_at is None or self._reader is None or self._failure is not None:
            return
        now = timestamp_s if timestamp_s is not None else time.monotonic()
        if not force and self._last_sample_at is not None and now - self._last_sample_at < self._interval_s:
            return
        try:
            watts = float(self._reader())
            if watts < 0:
                raise ValueError(f"received negative power value {watts}")
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            if self._required:
                raise RuntimeError(f"Power measurement failed: {self._failure}") from exc
            return
        self._samples.append(PowerSample(now, watts))
        self._last_sample_at = now

    def stop(self) -> PowerReport:
        if self._started_at is None:
            return PowerReport(self.capability, "unavailable", 0, 0.0, reason=self.capability.reason)
        ended_at = time.monotonic()
        self.sample(force=True, timestamp_s=ended_at)
        duration = max(0.0, ended_at - self._started_at)
        if self._failure:
            return PowerReport(self.capability, "failed", len(self._samples), duration, reason=self._failure)
        if not self._samples:
            return PowerReport(self.capability, "unavailable", 0, duration, reason=self.capability.reason)
        average, peak, energy = summarize_power_samples(self._samples, duration)
        return PowerReport(self.capability, "measured", len(self._samples), duration, average, peak, energy)


def summarize_power_samples(samples: list[PowerSample], duration_s: float) -> tuple[float, float, float]:
    """Return time-weighted mean W, peak W, and energy J from samples."""

    if not samples:
        raise ValueError("at least one power sample is required")
    ordered = sorted(samples, key=lambda sample: sample.timestamp_s)
    peak = max(sample.watts for sample in ordered)
    if len(ordered) == 1:
        average = ordered[0].watts
        return average, peak, average * duration_s
    energy = sum(
        (previous.watts + current.watts) / 2 * (current.timestamp_s - previous.timestamp_s)
        for previous, current in zip(ordered, ordered[1:])
    )
    observed_duration = ordered[-1].timestamp_s - ordered[0].timestamp_s
    average = energy / observed_duration if observed_duration > 0 else ordered[-1].watts
    if duration_s > observed_duration:
        energy += average * (duration_s - observed_duration)
    return average, peak, energy


def _reader_for(capability: PowerCapability, device: str, sensor_path: str | None) -> Callable[[], float]:
    if capability.source == "nvml":
        return _nvml_reader(device)
    if capability.source == "tegrastats":
        return _tegrastats_reader
    if capability.source == "powermetrics":
        return _powermetrics_reader
    if capability.source == "sysfs":
        path = Path(sensor_path) if sensor_path else _raspberry_power_paths()[0]
        return lambda: _read_sysfs_power(path)
    raise ValueError(f"unsupported power source {capability.source!r}")


def _nvml_reader(device: str) -> Callable[[], float]:
    import pynvml

    pynvml.nvmlInit()
    index = _cuda_index(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    return lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def _tegrastats_reader() -> float:
    process = subprocess.Popen(
        ["tegrastats", "--interval", "100"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        output, errors = process.communicate(timeout=0.3)
    except subprocess.TimeoutExpired:
        process.terminate()
        output, errors = process.communicate(timeout=1)
    if process.returncode not in (0, -15):
        raise RuntimeError(errors.strip() or f"tegrastats exited with status {process.returncode}")
    watts = _parse_tegrastats_watts(output)
    if watts is None:
        raise RuntimeError("tegrastats output did not contain VDD_IN power")
    return watts


def _parse_tegrastats_watts(output: str) -> float | None:
    matches = re.findall(r"VDD_IN\s+(\d+)mW", output)
    return int(matches[-1]) / 1000.0 if matches else None


def _powermetrics_reader() -> float:
    output = subprocess.run(
        ["sudo", "-n", "powermetrics", "--samplers", "smc", "-i", "100", "-n", "1"],
        capture_output=True, text=True, check=True, timeout=5,
    ).stdout
    watts = _parse_powermetrics_watts(output)
    if watts is None:
        raise RuntimeError("powermetrics output did not contain a power reading")
    return watts


def _parse_powermetrics_watts(output: str) -> float | None:
    for label in ("System Power", "Combined Power", "Package Power"):
        match = re.search(rf"{label}:?\s+([\d.]+)\s*(mW|W)", output, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value / 1000.0 if match.group(2).lower() == "mw" else value
    return None


def _read_sysfs_power(path: Path) -> float:
    value = float(path.read_text().strip())
    # Linux hwmon's power*_input is commonly microwatts. Permit a W value
    # for custom sensor bridges by treating values under 1,000 as watts.
    return value / 1_000_000.0 if value >= 1_000 else value


def _raspberry_power_paths() -> list[Path]:
    paths = list(Path("/sys/class/hwmon").glob("hwmon*/power*_input"))
    paths.extend(Path("/sys/bus/iio/devices").glob("iio:device*/in_power*_input"))
    return [path for path in paths if path.is_file()]


def _cuda_index(device: str) -> int:
    if ":" not in device:
        return 0
    return int(device.split(":", maxsplit=1)[1])


def _is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists()


def _is_raspberry_pi() -> bool:
    model = Path("/proc/device-tree/model")
    try:
        return "raspberry pi" in model.read_text(errors="ignore").lower()
    except OSError:
        return False


def _power_unavailable_message(capability: PowerCapability) -> str:
    detail = capability.reason or "No supported power source is available."
    hint = f" {capability.setup_hint}" if capability.setup_hint else ""
    return f"Power measurement is required but unavailable: {detail}{hint}"
