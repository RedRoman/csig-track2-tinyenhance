"""CPU-only fetch of actual committed D full outputs; no model/GPU launch."""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
VAL = OUT.parent / 'nr_push4_val_visuals_20261014' / 'diagnostic.py'
LABEL = 'push4_d7500_compiled_all'
DIRECTORY = '/cache/independent_sr_cuda_20261013/nr_push4_full_evaluations/' + LABEL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--timeout-seconds', type=int, default=2100)
    p.add_argument('--interval-seconds', type=int, default=60)
    a = p.parse_args()
    assert re.fullmatch('[0-9a-f]{64}', a.checkpoint_sha256)
    assert 30 <= a.interval_seconds <= 120 and 60 <= a.timeout_seconds <= 3600
    log = OUT / 'fetch_D_watch.log'
    start = time.monotonic(); last_build = None

    def run(args):
        r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
        item = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    args=list(map(str, args)), returncode=r.returncode,
                    stdout=r.stdout.strip(), stderr=r.stderr.strip())
        with log.open('a') as stream:
            stream.write(json.dumps(item) + '\n'); stream.flush()
        return r.returncode

    while time.monotonic() - start <= a.timeout_seconds:
        if run([OUT / 'gallery.py', 'fetch', '--label', LABEL, '--directory', DIRECTORY,
                '--checkpoint-sha256', a.checkpoint_sha256, '--name', 'D7500 / M2.5 reward0.075', '--allow-progress']) == 0:
            state = json.loads((OUT / 'sources.json').read_text())
            model = next(m for m in state['models'] if m['label'] == LABEL)
            complete = model['final_manifest'] and model['full_quality'] is not None and set(model['sources']) == {'7', '16', '41', '83'}
            if model['committed_full_render_count'] >= 3:
                val_code = run([VAL, '--additional-label', LABEL, '--additional-checkpoint-sha256', a.checkpoint_sha256] + (['--final'] if complete else []))
            else:
                val_code = 1
            build_state = (tuple(model['sources']), model['final_manifest'], model['full_quality'] is not None)
            if model['sources'] and build_state != last_build:
                assert run([OUT / 'gallery.py', 'build']) == 0
                assert run([OUT / 'gallery.py', 'audit']) == 0
                last_build = build_state
            if complete and val_code == 0:
                print(json.dumps(dict(status='PASS', full103_418_and_four_D_originals=True,
                                      fixed_val1_five_panes_final=True, selection_bound=False)))
                return
        time.sleep(a.interval_seconds)
    raise SystemExit('Timed out waiting for complete actual D full outputs; no final selection was inferred')


if __name__ == '__main__':
    main()
