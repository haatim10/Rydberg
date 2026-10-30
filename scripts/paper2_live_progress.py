"""Read-only, localhost live view of the existing Paper 2 experiment."""
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import time

DATA = Path(__file__).resolve().parents[1] / 'results/paper2_long_training'


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def page():
    status = read_json(DATA / 'status.json', {})
    name = status.get('run', '')
    state = status.get('state', 'Waiting for status')
    active = name.split('_')
    label = (f'Seed {active[0][4:]} · {"With Hankel structure" if active[1] == "H" else "Without Hankel structure"}'
             if len(active) == 3 else state.replace('_', ' ').title())
    age = max(0, time.time() - (DATA / 'status.json').stat().st_mtime)
    updated = datetime.fromtimestamp((DATA / 'status.json').stat().st_mtime).astimezone().strftime('%H:%M:%S %Z')
    histories = {p.parent.name: read_json(p, []) for p in DATA.glob('seed*/history.json')}
    complete = len(list(DATA.glob('seed*/result.json')))
    epochs_done = sum(len(h) for h in histories.values())
    batch = status.get('batch', 0)
    batches = status.get('batches', 1)
    epoch_fraction = min(1, batch / max(1, batches)) if state == 'training' else 0
    current_saved = len(histories.get(name, []))
    extra = epoch_fraction if status.get('epoch', 0) > current_saved else 0
    total_percent = min(100, 100 * (epochs_done + extra) / 546)
    rows = []
    for seed in (1, 2, 3):
        for arm, title in [('U', 'Without structure'), ('H', 'With structure')]:
            cells = []
            for budget in (13, 26, 52):
                run = f'seed{seed}_{arm}_{budget}ep'
                count = len(histories.get(run, []))
                finished = (DATA / run / 'result.json').exists()
                text = 'Complete' if finished else f'{count} / {budget} epochs' if count else 'Queued'
                if run == name and state == 'training':
                    text = f'Epoch {status["epoch"]} / {budget}'
                kind = 'active' if run == name and state == 'training' else 'done' if finished else ''
                cells.append(f'<td class="{kind}">{escape(text)}</td>')
            rows.append(f'<tr><td>{seed}</td><td>{title}</td>{"".join(cells)}</tr>')
    curve = histories.get(name, [])
    recent = ''.join(f'<tr><td>{h["epoch"]}</td><td>{h["train_loss"]:.5f}</td>'
                     f'<td>{h["validation_db"]:.3f} dB</td><td>{h["seconds"]/60:.1f} min</td></tr>'
                     for h in reversed(curve[-10:]))
    if not recent:
        recent = '<tr><td colspan="4">Appears after the first complete epoch of this run.</td></tr>'
    badge = 'Training' if state == 'training' else state.replace('_', ' ').title()
    note = f'Last training update {updated} · {int(age)} seconds ago'
    if age > 600 and state != 'complete':
        note += ' · No recent update; the Mac may be asleep or the process may need checking.'
    error = '<p class="error">' + escape(str(status.get('error', ''))) + '</p>' if state == 'failed' else ''
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="5"><title>Paper 2 — Live training</title>
<style>
:root{{color-scheme:light dark;--bg:light-dark(#f6f7f9,#11161d);--panel:light-dark(#fff,#1a222d);--ink:light-dark(#172432,#edf2f8);--muted:light-dark(#516174,#abb9c9);--line:light-dark(#e0e6ed,#334151);--accent:light-dark(#126956,#66d7b1)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:960px;padding:32px 24px;margin:auto}}h1{{font-size:28px;margin:0 0 4px}}h2{{font-size:19px;margin:0 0 10px}}
p{{margin:8px 0}}.muted{{color:var(--muted)}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin:22px 0}}
.top{{display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}}.badge{{color:var(--accent);font-weight:600}}
.epoch{{font-size:38px;font-weight:600;margin:12px 0 0;font-variant-numeric:tabular-nums}}
progress{{accent-color:var(--accent);width:100%;height:14px;margin:10px 0}}.small{{font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:11px 7px;border-bottom:1px solid var(--line)}}th{{font-weight:500;color:var(--muted)}}
td.active{{color:var(--accent);font-weight:650}}td.done{{color:var(--accent)}}.tablewrap{{overflow-x:auto}}.error{{white-space:pre-wrap}}
@media(max-width:520px){{main{{padding:20px 12px}}.panel{{padding:15px}}h1{{font-size:24px}}.epoch{{font-size:30px}}}}
</style></head><body><main>
<div class="top"><div><h1>Paper 2 · Live training</h1><p class="muted">Running on your MacBook Air · Apple M2 CPU</p></div><div class="badge">{escape(badge)}</div></div>
<section class="panel" aria-label="Current training run"><h2>{escape(label)}</h2>
<div class="epoch">Epoch {status.get('epoch', '—')} <span class="muted">/ {status.get('budget', '—')}</span></div>
<progress max="100" value="{100*epoch_fraction:.1f}" aria-label="Current epoch progress"></progress>
<div class="top"><span>Batch {batch:,} / {batches:,}</span><span>{100*epoch_fraction:.0f}% of this epoch</span></div>
<p class="muted small">{escape(note)}</p>{error}</section>
<section class="panel"><h2>Whole experiment</h2><div class="top"><span>{complete} / 18 runs complete</span><span>{total_percent:.1f}% of planned training epochs</span></div>
<progress max="100" value="{total_percent:.2f}" aria-label="Whole experiment progress"></progress>
<p class="muted small">{epochs_done} complete epochs out of 546. Longer runs count for more of the total work.</p>
<div class="tablewrap"><table><thead><tr><th>Seed</th><th>Model</th><th>13 epochs</th><th>26 epochs</th><th>52 epochs</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="panel"><h2>Recent completed epochs · current run</h2>
<div class="tablewrap"><table><thead><tr><th>Epoch</th><th>Training loss</th><th>Validation error</th><th>Duration</th></tr></thead><tbody>{recent}</tbody></table></div>
<p class="muted small">Training loss is mean normalized squared error. Validation error is in dB. Lower is better for both.</p></section>
<p class="muted small">Refreshes automatically every 5 seconds. The training process writes progress every 100 batches. Closing this view does not stop training.</p>
</main></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/', '/index.html'):
            self.send_error(404)
            return
        try:
            body = page().encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    print('Paper 2 live progress: http://127.0.0.1:8766', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8766), Handler).serve_forever()
