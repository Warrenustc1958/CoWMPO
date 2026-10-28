"""S5 debug: all Grounding DINO candidates for given prompts on one evidence frame, saved as an overlay."""
import sys
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
M = "/data_cf/cowmpo/builder_assets/models/grounding-dino-base"
ep, out = sys.argv[1], sys.argv[2]
prompts = sys.argv[3].split("|")
gp = AutoProcessor.from_pretrained(M); gm = AutoModelForZeroShotObjectDetection.from_pretrained(M).cuda().eval()
img = Image.open(f"/data_cf/cowmpo/evidence/libero_goal_dev3/{ep}/frame0.png").convert("RGB")
big = img.resize((768, 768), Image.BICUBIC)
canvas = img.resize((512, 512)); dr = ImageDraw.Draw(canvas)
cols = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "blue"]
for i, pr in enumerate(prompts):
    inp = gp(images=big, text=pr + ".", return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = gm(**inp)
    r = gp.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.1, text_threshold=0.1, target_sizes=[(256, 256)])[0]
    for s, b in sorted(zip(r["scores"].tolist(), r["boxes"].tolist()), reverse=True)[:5]:
        print(pr, round(s, 3), [round(x) for x in b])
        dr.rectangle([x * 2 for x in b], outline=cols[i % len(cols)], width=2)
        dr.text((b[0] * 2 + 2, b[1] * 2 + 2), f"{pr[:10]} {s:.2f}", fill=cols[i % len(cols)])
canvas.save(out)
