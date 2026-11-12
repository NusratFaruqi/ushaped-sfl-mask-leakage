"""Builds a Kaggle kernel folder for one tier (does NOT push it).
  python kaggle/make_kernel.py 0 <kaggle-username>   ->  kaggle/build/tier0/
Then, when you are ready:  kaggle kernels push -p kaggle/build/tier0
The code is embedded as a base64 zip so no separate dataset upload is needed."""
import base64, io, json, os, sys, zipfile
tier, user = sys.argv[1], sys.argv[2]
after = sys.argv[3].split(",") if len(sys.argv) > 3 else []      # earlier tiers whose runs/ this tier needs
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for d, _, fs in os.walk(root):
        if "build" in d or "__pycache__" in d:
            continue
        for f in fs:
            if f.endswith(".py"):
                z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), root))
sys.path.insert(0, os.path.join(root, "kaggle")); from plan import T
cmds = T[tier]
script = f'''import base64, io, zipfile, subprocess, os, shutil
os.makedirs("/kaggle/working/code", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{base64.b64encode(buf.getvalue()).decode()}"))).extractall("/kaggle/working/code")
os.chdir("/kaggle/working/code")
subprocess.run("pip install -q opacus lpips", shell=True)
import glob, time, json as _j
for prev in glob.glob("/kaggle/input/**/runs", recursive=True):   # outputs of earlier tiers
    shutil.copytree(prev, "/kaggle/working/runs", dirs_exist_ok=True); print("restored", prev, flush=True)
cmds, status, T0 = {json.dumps(cmds)}, [], time.time()
import threading, queue, torch
SETUP = ("python check_data.py", "python train.py --equiv-check", "python train.py --prepare-splits")
os.makedirs("/kaggle/working/logs", exist_ok=True)
def fix(c):
    if c.startswith("python train.py") and "--equiv" not in c and "--out" not in c:
        c += " --out /kaggle/working/runs"
    return c
def run(i, c, gpu):
    t = time.time(); log = f"/kaggle/working/logs/{{i:03d}}.log"
    print(f"[{{time.strftime('%H:%M')}}] START {{i}}/{{len(cmds)}} gpu{{gpu}}: {{c}}", flush=True)
    with open(log, "w") as fh:
        r = subprocess.run(c, shell=True, stdout=fh, stderr=subprocess.STDOUT, env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}})
    tail = open(log).read().splitlines()[-25:]
    s = {{"i": i, "cmd": c, "exit": r.returncode, "min": round((time.time() - t) / 60, 1)}}
    status.append(s); _j.dump(status, open("/kaggle/working/status.json", "w"), indent=1)
    print(f"[{{time.strftime('%H:%M')}}] END   {{i}}/{{len(cmds)}} exit {{r.returncode}} in {{s['min']}} min (elapsed {{(time.time() - T0) / 3600:.2f}} h)", flush=True)
    print("    " + "\\n    ".join(tail[-(25 if r.returncode else 4):]), flush=True)
    return r.returncode
todo = [(i, fix(c)) for i, c in enumerate(cmds, 1)]
setup = [x for x in todo if x[1].startswith(SETUP)]; rest = [x for x in todo if x not in setup]
ok = True
for i, c in setup:
    if run(i, c, 0) and c.startswith("python check_data.py"):
        print("DATA CHECK FAILED: stopping", flush=True); ok = False; break
if ok:
    q = queue.Queue(); [q.put(x) for x in rest]
    def worker(g):
        while True:
            try: i, c = q.get_nowait()
            except queue.Empty: return
            run(i, c, g)
    ths = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, torch.cuda.device_count()))]
    [t.start() for t in ths]; [t.join() for t in ths]
bad = [s for s in status if s["exit"]]
print("\\nSUMMARY:", len(status) - len(bad), "ok,", len(bad), "failed of", len(cmds), flush=True)
for s in bad: print("FAILED:", s["cmd"], flush=True)
'''
out = os.path.join(root, "kaggle", "build", f"tier{tier}"); os.makedirs(out, exist_ok=True)
open(os.path.join(out, "run.py"), "w").write(script)
json.dump({"id": f"{user}/sfl-unet-cut-tier{tier}", "title": f"sfl-unet-cut-tier{tier}", "code_file": "run.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
           "enable_internet": True,
           "kernel_sources": [(t.split("/")[0] if "/" in t else user) + f"/sfl-unet-cut-tier{t.split('/')[-1]}".lower() for t in after],   # "owner/tier" = a notebook shared from another account
           "dataset_sources": ["gokulraj324/polypgen20021-video", "balraj98/cvcclinicdb", "nguyenvoquocduong/etis-laribpolypdb",
                               "debeshjha1/kvasirseg", "dtruon46/bkai-igh-neopolyp"]},
          open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
print("built", out, len(cmds), "commands")
