#!/usr/bin/env python
"""Local live dashboard for the reco-net training runs.

Parses tmp_suite/*.log (written by train_organism.py via the run_*.sh queues),
groups runs by experiment, and serves a self-refreshing web page with progress
bars + per-category evals + sparklines. Stdlib only (no deps, read-only).

Run:    .venv/bin/python experiments/dashboard.py [--port 8765]
Open:   http://localhost:8765
Stop:   Ctrl-C (or kill the background process)
"""
import os, re, json, glob, argparse, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(REPO, "tmp_suite")

# d152 all-cats reference (winner recipe) = the performance targets we compare against.
TARGET = {"all": 0.891, "cat0": 0.456, "cat1": 0.472, "cat2": 0.410,
          "cat3": 0.602, "cat4": 0.960, "cat5": 0.936}
# experiment groups (prefix before first "_"); order = display order.
GROUPS = [
    ("cvg",  "G1c · convergence sweep — boosted-only, 300 ep"),
    ("g1c",  "G1c · cat5 probe — boosted-only"),
    ("minB", "G1c · initial grid — boosted-only, 30 ep"),
    ("a2",   "A2 · free-lunch probe — d152, all categories"),
]
GROUP_LABEL = dict(GROUPS)
GROUP_ORDER = {g: i for i, (g, _) in enumerate(GROUPS)}

PAR_RE    = re.compile(r"parameters: ([\d,]+)")
OUT_RE    = re.compile(r"^out: (.+)$", re.M)
EPLOSS_RE = re.compile(r"epoch (\d+): train_loss=([\d.]+)")
STEP_RE   = re.compile(r"ep (\d+) step (\d+)/(\d+) loss [\d.eE+-]+ \((\d+)s\)")
DONE_RE   = re.compile(r"done in ([\d.]+) min")
CATS = ["all", "cat0", "cat1", "cat2", "cat3", "cat4", "cat5"]
# metric keys are val/PerfectRecoPct_all (overall) and ..._all_cat{N} (per category)
KEYSUF = {"all": "all", "cat0": "all_cat0", "cat1": "all_cat1", "cat2": "all_cat2",
          "cat3": "all_cat3", "cat4": "all_cat4", "cat5": "all_cat5"}


def _catlist(txt, key):
    return [float(x) for x in re.findall(
        r"'val/PerfectRecoPct_%s': tensor\(([\d.eE+-]+)\)" % re.escape(key), txt)]


def parse_log(path):
    txt = open(path, errors="ignore").read()
    tag = os.path.basename(path)[:-4]
    d = {"tag": tag, "group": tag.split("_")[0]}
    d["group_label"] = GROUP_LABEL.get(d["group"], d["group"])
    m = PAR_RE.search(txt)
    d["params"] = int(m.group(1).replace(",", "")) if m else None
    cfg = {}
    mo = OUT_RE.search(txt)
    if mo:
        cj = os.path.join(mo.group(1).strip(), "config.json")
        if os.path.exists(cj):
            try: cfg = json.load(open(cj))
            except Exception: pass
    d["d_model"], d["blocks"], d["heads"] = cfg.get("d_model"), cfg.get("num_blocks"), cfg.get("num_heads")
    d["mlp"], d["keep_cats"] = cfg.get("include_mlp"), cfg.get("keep_cats")
    total = cfg.get("num_epochs")
    cats = {k: _catlist(txt, KEYSUF[k]) for k in CATS}
    d["cats"] = {k: (v[-1] if v else None) for k, v in cats.items()}
    d["best"] = {k: (max(v) if v else None) for k, v in cats.items()}
    d["spark"] = cats["cat5"][-60:] if (d["keep_cats"] == [4, 5] and cats["cat5"]) else cats["all"][-60:]
    # cat4 AND cat5 are equally important for boosted runs (matching d152 means BOTH hit
    # target); all-cats runs headline on "all". Provide cat-neutral ranking + trend fields.
    rel = ["cat4", "cat5"] if d["keep_cats"] == [4, 5] else ["all"]
    d["rel_cats"] = rel
    d["sparks"] = {c: cats[c][-80:] for c in rel}
    d["margins"] = {c: (d["cats"][c] - TARGET[c]) if d["cats"][c] is not None else None for c in rel}
    _mv = [m for m in d["margins"].values() if m is not None]
    d["key_margin"] = min(_mv) if _mv else None      # margin of the BINDING (worst) relevant category
    d["key_cat"] = (min(d["margins"], key=lambda c: d["margins"][c] if d["margins"][c] is not None else 9)
                    if _mv else None)
    losses = EPLOSS_RE.findall(txt)
    completed = len(losses)
    steps = STEP_RE.findall(txt)
    if steps:
        e, s, t, el = steps[-1]
        d["cur_step"], d["tot_steps"], d["elapsed_s"] = int(s), int(t), int(el)
        cur_ep = max(int(e), completed)
    else:
        d["cur_step"] = d["tot_steps"] = d["elapsed_s"] = 0
        cur_ep = completed
    d["cur_epoch"], d["total_epochs"] = cur_ep, total
    dn = DONE_RE.search(txt)
    mtime = os.path.getmtime(path)
    if dn:
        d["status"], d["duration_min"] = "done", float(dn.group(1))
    elif time.time() - mtime < 180:
        d["status"] = "running"
    else:
        d["status"] = "stalled"
    if d["elapsed_s"] and cur_ep > 0:
        frac = cur_ep + (d["cur_step"] / d["tot_steps"] if d["tot_steps"] else 0)
        spe = d["elapsed_s"] / max(frac, 1e-9)
        d["sec_per_epoch"] = round(spe, 1)
        if total:
            d["eta_min"] = round(max(total - cur_ep, 0) * spe / 60, 1)
    return d


def collect():
    runs = []
    for p in sorted(glob.glob(os.path.join(LOGDIR, "*.log"))):
        try:
            r = parse_log(p)
        except Exception:
            continue
        if r["params"] is None and r["cur_epoch"] == 0:
            continue  # not a real training log (dashboard.log, driver output, etc.)
        runs.append(r)
    runs.sort(key=lambda r: (GROUP_ORDER.get(r["group"], 99),
                             r.get("d_model") or 0, r.get("blocks") or 0, r["tag"]))
    return {"runs": runs, "target": TARGET, "now": time.time()}


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>reco-net training</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#c9d1d9;--mut:#8b949e;
--grn:#3fb950;--amb:#d29922;--red:#f85149;--blu:#58a6ff;--track:#21262d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{position:sticky;top:0;background:#0d1117ee;backdrop-filter:blur(6px);
border-bottom:1px solid var(--bd);padding:12px 18px;display:flex;align-items:baseline;gap:16px;z-index:5}
h1{font-size:16px;margin:0;font-weight:600}#summary{color:var(--mut);font-size:13px}
#updated{margin-left:auto;color:var(--mut);font-size:12px}
.grp{margin:18px}.grp h2{font-size:13px;font-weight:600;color:var(--blu);
margin:0 0 10px;padding-bottom:6px;border-bottom:1px solid var(--bd);letter-spacing:.02em}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:9px;padding:12px 14px}
.row1{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.tag{font-weight:600;font-size:13px}.chips{margin-left:auto;display:flex;gap:5px;flex-wrap:wrap}
.chip{font:11px ui-monospace,Menlo,monospace;color:var(--mut);background:var(--track);
border-radius:5px;padding:1px 6px}
.badge{font-size:10px;text-transform:uppercase;letter-spacing:.05em;padding:2px 7px;border-radius:10px;font-weight:700}
.b-running{color:#0d1117;background:var(--grn)}.b-done{color:#0d1117;background:var(--blu)}
.b-stalled{color:#0d1117;background:var(--amb)}
.prog{height:7px;background:var(--track);border-radius:4px;overflow:hidden;margin:7px 0 3px}
.prog>div{height:100%;background:var(--blu);border-radius:4px;transition:width .4s}
.pmeta{display:flex;justify-content:space-between;color:var(--mut);font:11px ui-monospace,monospace}
.evals{margin-top:10px;display:flex;flex-direction:column;gap:6px}
.ev{display:grid;grid-template-columns:42px 1fr 96px;align-items:center;gap:8px}
.ev .lab{font:11px ui-monospace,monospace;color:var(--mut)}
.bar{position:relative;height:14px;background:var(--track);border-radius:3px;overflow:hidden}
.bar>.fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.bar>.tgt{position:absolute;top:-2px;bottom:-2px;width:2px;background:#fff8}
.ev .num{font:11px ui-monospace,monospace;text-align:right;white-space:nowrap}
.ev .num b{font-size:12px}.ev .num span{color:var(--mut)}
.spark{margin-top:9px}.foot{margin-top:8px;color:var(--mut);font:11px ui-monospace,monospace;display:flex;justify-content:space-between}
.empty{color:var(--mut);padding:40px;text-align:center}
</style></head><body>
<header><h1>reco-net training</h1><span id=summary></span><span id=updated></span></header>
<div id=root></div>
<script>
const TGT_LO=0.80, TGT_HI=1.0;  // zoom the eval bars onto the high-value band
let TARGET={};
const pct=v=>Math.max(0,Math.min(100,(v-TGT_LO)/(TGT_HI-TGT_LO)*100));
function color(v,t){ if(v==null)return'var(--mut)'; if(v>=t)return'var(--grn)'; if(v>=t-0.02)return'var(--amb)'; return'var(--red)'; }
function spark(vals){
  if(!vals||vals.length<2)return'';
  const w=300,h=30,n=vals.length,mn=Math.min(...vals),mx=Math.max(...vals),rng=(mx-mn)||1;
  const pts=vals.map((v,i)=>`${(i/(n-1)*w).toFixed(1)},${(h-2-(v-mn)/rng*(h-4)).toFixed(1)}`).join(' ');
  return `<svg class=spark width=100% viewBox="0 0 ${w} ${h}" preserveAspectRatio=none>
   <polyline points="${pts}" fill=none stroke="var(--blu)" stroke-width=1.5/></svg>`;
}
function evalRow(k,latest,best){
  const t=TARGET[k]; const c=color(latest,t);
  const num = latest==null?'<span>—</span>':`<b style="color:${c}">${latest.toFixed(3)}</b> <span>/${best!=null?best.toFixed(3):'—'}</span>`;
  const fill = latest==null?'':`<div class=fill style="width:${pct(latest)}%;background:${c}"></div>`;
  const tgt = t!=null?`<div class=tgt style="left:${pct(t)}%"></div>`:'';
  return `<div class=ev><div class=lab>${k}</div><div class=bar>${fill}${tgt}</div><div class=num>${num}</div></div>`;
}
function card(r){
  const arch=[r.d_model?'d'+r.d_model:'', r.blocks?'b'+r.blocks:'', r.heads?'h'+r.heads:'',
    (r.mlp===false?'attn-only':(r.mlp?'mlp':'')), r.params?(r.params.toLocaleString()+'p'):''].filter(Boolean);
  if(r.keep_cats) arch.push('keep'+r.keep_cats.join(''));
  const chips=arch.map(a=>`<span class=chip>${a}</span>`).join('');
  const tot=r.total_epochs||'?'; const ep=r.cur_epoch||0;
  const pp=r.total_epochs?Math.min(100,ep/r.total_epochs*100):0;
  const stepinfo=(r.status==='running'&&r.tot_steps)?`step ${r.cur_step}/${r.tot_steps}`:'';
  const keys = r.keep_cats&&r.keep_cats.join('')==='45' ? ['cat4','cat5']
             : ['all','cat0','cat1','cat2','cat3','cat4','cat5'];
  const evals = keys.map(k=>evalRow(k, r.cats[k], r.best[k])).join('');
  let foot='';
  if(r.status==='done') foot=`done · ${r.duration_min} min`;
  else if(r.status==='running') foot=`${r.sec_per_epoch?r.sec_per_epoch+'s/ep':''}${r.eta_min!=null?' · ETA '+r.eta_min+'m':''}`;
  else foot='stalled (no recent output)';
  return `<div class=card>
    <div class=row1><span class=tag>${r.tag}</span><span class=chips>${chips}</span></div>
    <span class="badge b-${r.status}">${r.status}</span>
    <div class=prog><div style="width:${pp}%"></div></div>
    <div class=pmeta><span>ep ${ep}/${tot}</span><span>${stepinfo}</span><span>${pp.toFixed(0)}%</span></div>
    <div class=evals>${evals}</div>
    ${spark(r.spark)}
    <div class=foot><span>${foot}</span><span>${keys.length>2?'all+cat0-5 (latest/best)':'cat4,cat5 (latest/best)'}</span></div>
  </div>`;
}
async function refresh(){
  let data; try{ data=await (await fetch('/data')).json(); }catch(e){ return; }
  TARGET=data.target;
  const runs=data.runs, root=document.getElementById('root');
  const groups={};
  for(const r of runs){ (groups[r.group_label] ??= []).push(r); }
  let html='';
  for(const label in groups){
    const gs=groups[label], run=gs.filter(r=>r.status==='running').length, done=gs.filter(r=>r.status==='done').length;
    html+=`<div class=grp><h2>${label} &nbsp;·&nbsp; ${gs.length} runs · ${run} running · ${done} done</h2>
      <div class=cards>${gs.map(card).join('')}</div></div>`;
  }
  root.innerHTML = html || '<div class=empty>No runs found in tmp_suite/ yet.</div>';
  const nr=runs.filter(r=>r.status==='running').length, nd=runs.filter(r=>r.status==='done').length;
  document.getElementById('summary').textContent=`${runs.length} runs · ${nr} running · ${nd} done`;
  document.getElementById('updated').textContent='updated '+new Date().toLocaleTimeString();
}
refresh(); setInterval(refresh, 4000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/data"):
            body = json.dumps(collect()).encode(); ctype = "application/json"
        elif path in ("/", "/index.html"):
            body = PAGE.encode(); ctype = "text/html; charset=utf-8"
        else:
            # serve alternative designs from experiments/dashboards/<name>.html at /<name>
            name = path.strip("/")
            fpath = os.path.join(REPO, "experiments", "dashboards", name + ".html")
            if re.match(r"^[\w-]+$", name) and os.path.isfile(fpath):
                body = open(fpath, "rb").read(); ctype = "text/html; charset=utf-8"
            else:
                self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"reco-net dashboard → http://localhost:{args.port}   (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
