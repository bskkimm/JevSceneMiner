"""Run with the installed wheel's Python -I from outside the source directory."""
import json
from importlib.resources import files

from jevsceneminer.cli import DEFAULT_LABELS
from jevsceneminer.jev import load_labels
from jevsceneminer.viewer import PAGE

labels = load_labels(DEFAULT_LABELS, traffic_side='right', has_indicator=False)
assert labels.version == 'jsm-1.4'
assert len(labels.lateral) == 11 and len(labels.longitudinal) == 5
assert PAGE.is_file()
assert all(PAGE.with_name(name).is_file() for name in ['camera_decisions.js', 'bev_geometry.js', 'editor.js'])
assert files("jevsceneminer").joinpath("presets/nuplan-1hz.yaml").is_file()
print(json.dumps({'labels': labels.version, 'lateral_choices': 11, 'longitudinal_choices': 5,
                  'viewer_resources': 'available', 'preferred_preset': 'available'}))
