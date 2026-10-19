"""Summarize one-environment diagnostic traces, excluding stage-entry settling.

Usage: python scripts/summarize_little_white_motion.py TRACE.json [TRACE.json ...]
"""
import json
from pathlib import Path
import sys
import numpy as np


def summarize(path):
    data = json.loads(Path(path).read_text())
    frames, dt = data['frames'], data['control_dt']
    names = data['joint_names']
    result = {'trace': str(path), 'scope': 'first environment only', 'stages': {}}
    for stage, label in [(0, 'stand'), (1, 'sit'), (4, 'land')]:
        selected = []
        age = 0
        for i, f in enumerate(frames):
            if f['stage'] != stage or f['done']:
                age = 0
                continue
            age += 1
            if i == 0 or frames[i-1]['done'] or frames[i-1]['stage'] != stage:
                continue
            if stage != 1 and age*dt <= .5:
                continue
            selected.append((i, f))
        if not selected:
            result['stages'][label] = {'samples': 0}
            continue
        states = np.asarray([f['state'] for _, f in selected])
        dq = np.asarray([f['dq'] for _, f in selected])
        delta = np.asarray([np.asarray(f['targets'])-frames[i-1]['targets'] for i, f in selected])
        result['stages'][label] = {
            'samples': len(selected),
            'height_mean_m': float(states[:, 6].mean()),
            'body_angular_rms_rad_s': np.sqrt((states[:, 3:6]**2).mean(0)).tolist(),
            'joint_velocity_rms_rad_s': dict(zip(names, np.sqrt((dq**2).mean(0)).tolist())),
            'target_step_rms_rad': dict(zip(names, np.sqrt((delta**2).mean(0)).tolist())),
        }
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        print(json.dumps(summarize(path), indent=2))
