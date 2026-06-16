# Champion training on Google Colab

Set **`CHAMPION`** once (config slug, lowercase — e.g. `ahri`, `ezreal`). Everything else derives from it.

Configs and source code come from **git** (`git pull`). Only **checkpoints** stay off GitHub (`models/` is gitignored). Upload **clips** separately (zip).

## 0. Configure

```python
CHAMPION = "ahri"   # <-- change this only
REPO_URL = "https://github.com/YOUR_USER/League-Timer.git"  # your fork
```

## 1. Clone or pull repo

```python
CHAMPION = "ahri"
REPO_URL = "https://github.com/YOUR_USER/League-Timer.git"

import os
if not os.path.isdir("/content/League-Timer"):
    !git clone {REPO_URL} /content/League-Timer
else:
    %cd /content/League-Timer
    !git pull
%cd /content/League-Timer
```

You get `configs/{CHAMPION}.yaml`, training code, and `requirements.txt` from the repo.

## 2. Upload clips

`data/` is not on GitHub. Upload `{CHAMPION}_clips.zip` (see **Build clips zip** below), then:

```python
CHAMPION = "ahri"

!mkdir -p data/{CHAMPION}
!unzip -q {CHAMPION}_clips.zip -d data/{CHAMPION}/
# expect data/{CHAMPION}/clips/manifest.json + clips/frames/*.npy
```

## 3. Install (CUDA)

```python
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q numpy pyyaml tqdm opencv-python-headless scikit-learn
!pip install -q pytorchvideo fvcore iopath
```

## 4. Train

```python
CHAMPION = "ahri"

!python -m src.train.train --config configs/{CHAMPION}.yaml --device cuda
```

Checkpoint writes to `models/{CHAMPION}/best.pt` (~10 MB, local only — download it).

## 5. Download checkpoint

```python
CHAMPION = "ahri"

from google.colab import files
files.download(f"models/{CHAMPION}/best.pt")
```

Copy to your local repo (also gitignored):

```text
League-Timer/models/{CHAMPION}/best.pt
```

## After download (local machine)

From git you already have `configs/{CHAMPION}.yaml`. You still need locally for live inference:

- `configs/templates/{champion}_name.png` (gitignored — capture per resolution)

Then:

```bash
python -m src.infer.live --config configs/{CHAMPION}.yaml \
    --checkpoint models/{CHAMPION}/best.pt

python -m src.app.server
```

---

## Build clips zip (local, before Colab)

After `python -m src.dataset.build --config configs/{CHAMPION}.yaml`:

```bash
CHAMPION=ahri

(cd "data/${CHAMPION}" && zip -r -q "../../outputs/${CHAMPION}_clips.zip" clips)
```

Upload `outputs/{CHAMPION}_clips.zip` to Colab (~100 MB for Ahri).

---

## Notes

- **On GitHub:** `configs/*.yaml`, source, `scripts/colab/COLAB.md`
- **Not on GitHub:** `models/`, `data/`, `outputs/`, `configs/templates/`, videos
- Training time: ~15–45 min on a Colab T4.
- First run downloads X3D-S weights via `torch.hub` (~20 MB).
