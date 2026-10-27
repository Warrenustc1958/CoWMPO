"""CoWM-PO Code Agent runner (host side; plan §3.11 L-Agent + L-Verify).

  evidence (first frames + task sentences) -> LLM -> world_program.py -> [Builder sandbox] check_program
  -> report + agentview render -> LLM ... (<= rounds)
The LLM only sees the SDK doc, the evidence and verifier reports/renders. Compilation and checks run in the
sandbox (no network, no target files). Every round is archived.
  --seed-program P : round 0 uses P instead of asking the LLM (forces the repair loop to run)
  --mock           : no network; a scripted 'agent' returns the reference program on every LLM turn
  --no-images      : text only
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys

from prompts import SYSTEM, build_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cowm.world.registry import diff_programs, register  # noqa: E402
RUN_BUILDER = os.path.join(HERE, "..", "isolation", "run_builder.sh")
REF = os.path.join(HERE, "..", "world", "ref_goal_program.py")


def load_evidence(ev_dir, n_images):
    eps = sorted(d for d in os.listdir(ev_dir) if os.path.isfile(os.path.join(ev_dir, d, "meta.json")))
    langs, picks = [], []
    for ep in eps:
        lang = json.load(open(os.path.join(ev_dir, ep, "meta.json")))["language"]
        if lang not in langs:
            langs.append(lang)
            if len(picks) < n_images:
                picks.append((ep, lang))
    text = ("One shared tabletop scene is used by all of these tasks (task sentences, one per line):\n"
            + "\n".join(f"- {s}" for s in langs)
            + "\n\nAttached images: the first agentview camera frame of episodes of: "
            + "; ".join(f"[{i + 1}] {lang}" for i, (_, lang) in enumerate(picks))
            + ".\nObject placements differ between episodes and are re-estimated per episode; use the images "
              "for which objects exist, their rough sizes, materials and a typical layout.")
    imgs = [os.path.join(ev_dir, ep, "frame0.png") for ep, _ in picks]
    return text, imgs


def img_block(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def sandbox_check(program_path, out_dir):
    cmd = [RUN_BUILDER, "python", "-m", "cowm.world.check_program", "--program", program_path, "--out", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    rep_path = os.path.join(out_dir, "check_report.json")
    if os.path.isfile(rep_path):
        return json.load(open(rep_path))
    return {"ok": False, "stage": "sandbox", "errors": [proc.stdout[-1500:], proc.stderr[-1500:]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="output dir under /data_cf/cowmpo/worlds (sandbox-writable)")
    p.add_argument("--evidence", default="/data_cf/cowmpo/evidence/libero_goal_dev3")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--seed-program", default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--provider", default=None, help="CodeWorld-Safe llm_providers.yaml provider name")
    a = p.parse_args()

    ev_text, ev_imgs = load_evidence(a.evidence, 0 if a.no_images else a.n_images)
    state = {"images_ok": not a.no_images}
    if a.mock:
        def ask(system, user, images):
            return "```python\n" + open(REF).read() + "\n```"
    else:
        sys.path.insert(0, "/mnt/nfs/fjj/Code/CodeWorld-Safe")
        from toolkits.codeworld_safe.agent.llm_client import LLMClient, LLMError

        client = LLMClient(a.provider)

        def ask(system, user, images):
            if state["images_ok"] and images:
                content = [img_block(i) for i in images] + [{"type": "text", "text": user}]
                try:
                    return client.complete(system, content, role="planner", max_tokens=8000)
                except LLMError as e:
                    print(f"image request failed ({str(e)[:160]}); falling back to text-only", flush=True)
                    state["images_ok"] = False
            return client.complete(system, user, role="planner", max_tokens=8000)

    os.makedirs(a.run, exist_ok=True)
    ev_ids = [os.path.basename(os.path.dirname(i)) for i in ev_imgs]
    prev, report, prev_render, history, prev_vid = None, None, None, [], None
    for r in range(a.rounds):
        rdir = os.path.join(a.run, f"round{r:02d}")
        os.makedirs(rdir, exist_ok=True)
        prog = os.path.join(rdir, "world_program.py")
        if r == 0 and a.seed_program:
            code, source = open(a.seed_program).read(), "seed"
        else:
            user = build_prompt(ev_text, prev, report)
            images = list(ev_imgs) + ([prev_render] if prev_render else [])
            with open(os.path.join(rdir, "prompt.txt"), "w") as f:
                f.write(SYSTEM + "\n\n[images: " + ", ".join(images) + "]\n\n" + user)
            reply = ask(SYSTEM, user, images)
            with open(os.path.join(rdir, "reply.txt"), "w") as f:
                f.write(reply)
            code, source = extract_code(reply), ("mock" if a.mock else "llm")
        with open(prog, "w") as f:
            f.write(code)
        rep = sandbox_check(prog, os.path.join(rdir, "world"))
        vid = None
        if os.path.isfile(os.path.join(rdir, "world", "world.json")):
            vid = register(os.path.join(rdir, "world"), parent=prev_vid, source=f"{a.run}:round{r:02d}:{source}",
                           evidence_ids=ev_ids,
                           changed=diff_programs(prev, code) if prev is not None else {"initial": True},
                           report=rep, cost={"round": r})
            prev_vid = vid
        history.append({"round": r, "source": source, "ok": rep["ok"], "stage": rep.get("stage"),
                        "gates": rep.get("gates"), "world_version": vid,
                        "errors": rep.get("errors", []), "images_sent": state["images_ok"]})
        print(f"round {r} [{source}]: {'PASS' if rep['ok'] else 'FAIL[' + str(rep.get('stage')) + ']'} "
              f"{json.dumps(rep.get('errors', []))[:400]}", flush=True)
        if rep["ok"]:
            break
        prev = code
        report = json.dumps({k: rep.get(k) for k in ("stage", "errors", "physics", "usability")},
                            indent=1, default=str)
        prev_render = rep.get("render") if rep.get("render") and os.path.isfile(rep["render"]) else None
    with open(os.path.join(a.run, "history.json"), "w") as f:
        json.dump({"mock": a.mock, "provider": a.provider, "seed_program": a.seed_program,
                   "history": history}, f, indent=2)
    print("AGENT_BUILD", "ACCEPTED" if history and history[-1]["ok"] else "NOT_ACCEPTED", f"rounds={len(history)}")


if __name__ == "__main__":
    main()
