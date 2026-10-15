"""Snapshot this bounded original-tool recheck; no training or GPU calls."""
from pathlib import Path
import datetime
import json

ROOT = Path('/cache/independent_sr_cuda_20261013')
out = ROOT / 'metric_rechecks/20261014_original_tool'
rows = {}
for label in ['compiled6000', 'conservative6250']:
    p = out / label / 'status.json'
    rows[label] = json.loads(p.read_text()) if p.exists() else {'status':'not_started'}
state = 'complete' if all(x['status']=='complete' for x in rows.values()) else ('failed' if any(x['status']=='failed' for x in rows.values()) else 'running')
result = dict(status=state, updated_at_bjt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              scope='fresh original score_dir.py; both existing native103 PNG sets; no new training',
              out_root=str(out), labels=rows, physical_cuda_devices={'compiled6000':1,'conservative6250':0})
(ROOT/'results/status_nr_metric_recheck_20261014.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({label:{k:x.get(k) for k in ['status','completed','scalar_metrics_completed']} for label,x in rows.items()}))
