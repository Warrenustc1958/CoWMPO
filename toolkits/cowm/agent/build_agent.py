"""CoWM-PO Code Agent runner (host side; plan §3.11 L-Agent + L-Verify).

  evidence -> LLM -> world_program.py -> [Builder sandbox] check_program -> report -> LLM ... (<= rounds)
The LLM only sees the SDK doc, the evidence summary and verifier reports. Compilation and checks run in
the sandbox (no network, no libero). Every round is archived.
  --mock: no network; a scripted 'agent' returns a broken program first, then the reference program,
          to test the loop itself.
"""
import argparse
import json
import os
import re
import subprocess
import sys

from prompts import SYSTEM, build_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_BUILDER = os.path.join(HERE, "..", "isolation", "run_builder.sh")
REF = os.path.join(HERE, "..", "world", "ref_goal_program.py")

GOAL_EVIDENCE = """Target suite: LIBERO-goal style tabletop. One shared scene is used by all of these tasks
(task language only; no target files are available to you):
- open the middle drawer of the cabinet
- put the bowl on the stove
- put the wine bottle on top of the cabinet
- open the top drawer and put the bowl inside
- put the bowl on top of the cabinet
- push the plate to the front of the stove
- put the cream cheese in the bowl
- turn on the stove
- put the bowl on the plate
- put the wine bottle on the rack
Build the shared scene. Exact sizes, frictions, damping and placements will be identified later from
recorded robot trajectories, so declare uncertain values with w.p(...)."""


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def mock_agent():
    yield "```python\ndef build(w):\n    import os\n    w.table()\n```"
    yield "```python\n" + open(REF).read() + "\n```"


def sandbox_check(program_path, out_dir):
    # paths are passed as seen inside the sandbox (same absolute paths are bound back)
    cmd = [RUN_BUILDER, "python", "-m", "cowm.world.check_program", "--program", program_path, "--out", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    rep_path = os.path.join(out_dir, "check_report.json")
    if os.path.isfile(rep_path):
        return json.load(open(rep_path))
    return {"ok": False, "stage": "sandbox", "errors": [proc.stdout[-1500:], proc.stderr[-1500:]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="output dir under /data_cf/cowmpo/worlds (sandbox-writable)")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--provider", default=None, help="CodeWorld-Safe llm_providers.yaml provider name")
    a = p.parse_args()

    if a.mock:
        agent = mock_agent()
        ask = lambda system, user: next(agent)  # noqa: E731
    else:
        sys.path.insert(0, "/mnt/nfs/fjj/Code/CodeWorld-Safe")
        from toolkits.codeworld_safe.agent.llm_client import LLMClient

        client = LLMClient(a.provider)
        ask = lambda system, user: client.complete(system, user, role="planner", max_tokens=8000)  # noqa: E731

    os.makedirs(a.run, exist_ok=True)
    prev, report, history = None, None, []
    for r in range(a.rounds):
        rdir = os.path.join(a.run, f"round{r:02d}")
        os.makedirs(rdir, exist_ok=True)
        user = build_prompt(GOAL_EVIDENCE, prev, report)
        with open(os.path.join(rdir, "prompt.txt"), "w") as f:
            f.write(SYSTEM + "\n\n" + user)
        reply = ask(SYSTEM, user)
        with open(os.path.join(rdir, "reply.txt"), "w") as f:
            f.write(reply)
        prog = os.path.join(rdir, "world_program.py")
        with open(prog, "w") as f:
            f.write(extract_code(reply))
        rep = sandbox_check(prog, os.path.join(rdir, "world"))
        history.append({"round": r, "ok": rep["ok"], "stage": rep.get("stage"), "errors": rep.get("errors", [])})
        print(f"round {r}: {'PASS' if rep['ok'] else 'FAIL[' + str(rep.get('stage')) + ']'} "
              f"{json.dumps(rep.get('errors', []))[:300]}", flush=True)
        if rep["ok"]:
            break
        prev, report = open(prog).read(), json.dumps({k: rep.get(k) for k in ("stage", "errors", "physics")},
                                                    indent=1, default=str)
    with open(os.path.join(a.run, "history.json"), "w") as f:
        json.dump({"mock": a.mock, "provider": a.provider, "history": history}, f, indent=2)
    print("AGENT_BUILD", "ACCEPTED" if history and history[-1]["ok"] else "NOT_ACCEPTED", f"rounds={len(history)}")


if __name__ == "__main__":
    main()
