"""Compatibility alias; business logic lives in application.benchmark."""

import sys
from fabric_defect_hub.application import benchmark as _implementation

sys.modules[__name__] = _implementation
