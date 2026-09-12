"""Compatibility alias; workspace logic lives in application.workspace."""

import sys
from fabric_defect_hub.application import workspace as _implementation

sys.modules[__name__] = _implementation
