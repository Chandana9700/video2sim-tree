# video2sim-tree

**Video2Sim — Deciduous Tree | CSCI 5961 AI Capstone, Fall 2026**  
Saint Louis University · Chandana Rajashekhar

Converts a handheld smartphone video of an outdoor deciduous tree into a simulation-ready USDA asset using a six-stage pipeline: FFmpeg → COLMAP → DA3 → SAM3 → HoloScene → USDA. The final asset is compatible with NVIDIA Isaac Sim and any USD-compliant simulator.

> **Partner repository:** [Sidhvi16/video2sim-cup](https://github.com/Sidhvi16/video2sim-cup) — cup dataset (same pipeline, different object).

---

## Repository Structure

```
video2sim-tree/
├── src/
│   ├── da3/              # Depth Anything v3 module code
│   ├── sam3/             # SAM3 / SAM2 module code
│   └── holoscene/        # HoloScene reconstruction module
├── scripts/
│   ├── batch_reconstruct.bat   # Windows: FFmpeg + COLMAP automation
│   ├── run_da3.sh              # SLURM job: DA3 depth estimation
│   ├── run_sam3.sh             # SLURM job: SAM3 instance segmentation
│   ├── run_holoscene.sh        # SLURM job: HoloScene reconstruction
│   └── run.sh                  # HPC: chain all three SLURM jobs
├── docs/
│   ├── pipeline_architecture.png
│   └── holoscene_patch.md      # Required HoloScene code patch
├── experiments/
│   ├── configs/                # SLURM job configs and hyperparameters
│   ├── logs/                   # SLURM output logs
│   └── figures/                # Generated evaluation figures
├── reports/                    # Weekly progress reports
├── poster/                     # poster.pdf + PowerPoint source
├── report/                     # final_report.pdf + LaTeX source
├── data/                       # NOT committed — see Data section below
│   └── input/custom/tree/
│       ├── images/             # JPEG frames (FFmpeg output)
│       ├── sparse/0/           # COLMAP TXT model
│       ├── transforms.json     # DA3 output
│       ├── instance_mask/      # SAM3 output
│       └── prompts.txt         # SAM3 text prompts
├── environment_da3.yml         # Conda env for DA3 + SAM3
├── environment_holoscene.yml   # Conda env for HoloScene
├── requirements.txt            # Python dependencies (pip-compatible)
└── README.md
```

---

## Object

A single outdoor deciduous tree in spring foliage — light green leaves, dark branching structure, a mulched circular base, and red brick campus buildings in the background. Captured with a handheld smartphone under natural daylight. The tree presents a significantly harder reconstruction challenge than a rigid indoor object due to wind-induced leaf motion blur, large regions of uniform sky, and the repetitive texture of the surrounding brick buildings.

**SAM3 prompts used** (`prompts.txt`):
```
Single decidious tree.
Spring foliage.
Light green leaves.
Mulched base.
Red brick buildings.
```

**Output format:** The tree pipeline produces a `.usda` (USD ASCII) file, as opposed to binary `.usdc`.

---

## Prerequisites

### Local Machine (Windows 10/11)

| Tool | Version | Install path |
|------|---------|-------------|
| FFmpeg | 7.x | `03 FFMPEG\` or `03 FFMPEG\bin\` |
| COLMAP | 4.0.2 | `01 COLMAP\` or `01 COLMAP\bin\` |
| MeshLab | 2023+ | Standard system install |

Folder layout expected by `batch_reconstruct.bat`:
```
01 COLMAP\
02 VIDEOS\
03 FFMPEG\
04 SCENES\
batch_reconstruct.bat   ← place here, one level above the four folders
```

### HPC Cluster (SLU Libra or equivalent SLURM cluster)

- SLURM scheduler
- NVIDIA H100 NVL or A100 for DA3 (80 GB VRAM recommended)
- NVIDIA L40S or A100 for SAM3 and HoloScene (40+ GB VRAM)
- Conda (Miniconda or Anaconda)
- Hugging Face account with access to `GonzaloMG/marigold-e2e-ft-normals`

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Chandana9700/video2sim-tree.git
cd video2sim-tree
```

### 2. Create conda environments (HPC)

```bash
# DA3 + SAM3 environment
conda env create -f environment_da3.yml
conda activate da3_env

# HoloScene environment (must be compiled on a GPU compute node)
# SSH into a GPU node first, then:
conda env create -f environment_holoscene.yml
conda activate holoscene_env
```

### 3. Apply the HoloScene patch

Before running any HoloScene jobs, apply the instance mesh fallback patch. This is a required fix — without it the job crashes in post-processing even after successful training completes.

```bash
cd src/holoscene/training/
cp holoscene_train.py holoscene_train.py.bak

# Open holoscene_train.py and locate instance_meshes_post_pruning (~line 529).
# Replace:
#   assert os.path.exists(obj_i_mesh_path), f"mesh {obj_i} does not exist"
# With the fallback block documented in docs/holoscene_patch.md
```

The full replacement block and explanation are in `docs/holoscene_patch.md`.

### 4. Export your Hugging Face token

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Re-export this in every new shell session before submitting SLURM jobs.

---

## Running the Pipeline

### Stage 1 & 2 — Local (Windows): Frame Extraction + COLMAP

1. Place your video file in `02 VIDEOS\`.
2. Double-click `scripts\batch_reconstruct.bat` (or run from Command Prompt).
3. The script extracts frames at `EXTRACT_FPS=0.5` and runs COLMAP feature extraction, sequential matching, sparse mapping, and TXT export automatically.
4. After completion, open the COLMAP GUI and verify that camera frustums cover the full tree from multiple angles with no large gaps.

> **Note on the tree scene:** The large uniform sky background reduces SIFT keypoint density above the canopy. If registration rate drops below 80%, increase `EXTRACT_FPS` to 0.7–1.0 and re-run.

### Stage 2.5 — Local: Point Cloud Cleaning (MeshLab)

```
File > Import Mesh > sparse\0\points3D.ply
Filters > Cleaning and Repairing > Remove Isolated Pieces (wrt Diameter, 10% threshold)
Filters > Cleaning and Repairing > Remove Duplicate Vertices
Visual inspection: select and delete any floating sky-region debris
File > Export Mesh As > points3D_clean.ply
```

### Stage 3 — Transfer to HPC

```bash
# From Windows terminal (adjust username as needed)
scp -r .\images <username>@libra.slu.edu:~/video2sim-conda2/data/input/custom/tree/
scp -r .\sparse\0\*.txt <username>@libra.slu.edu:~/video2sim-conda2/data/input/custom/tree/sparse/0/
scp points3D_clean.ply <username>@libra.slu.edu:~/video2sim-conda2/
```

### Stage 4, 5, 6 — HPC: DA3 → SAM3 → HoloScene

Option A — run individually:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
cd ~/video2sim-conda2

sbatch scripts/run_da3.sh
# Wait for completion, then:
sbatch scripts/run_sam3.sh
# Wait for completion, then:
sbatch scripts/run_holoscene.sh
```

Option B — chain automatically with SLURM dependencies:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
cd ~/video2sim-conda2
bash scripts/run.sh
```

Monitor jobs:

```bash
squeue -u $USER
tail -f experiments/logs/da3_tree_<jobid>.log
tail -f experiments/logs/sam3_tree_<jobid>.log
tail -f experiments/logs/holoscene_tree_<jobid>.log
```

Successful HoloScene training produces lines like:
```
38%|███▊      | 98/256 [15:20<22:01,  8.36s/it]
...
surface_100_whole.ply save to ./exps/holoscene_tree/.../plots
```

### Stage 7 — USDA Validation

Download the USDA file:

```bash
scp <username>@libra.slu.edu:~/video2sim-conda2/repo/modules/holoscene/exps/holoscene_tree/*/plots/*.usda .
```

Open in NVIDIA Isaac Sim: `File > Open > select .usda`. Verify:
- `metersPerUnit = 0.01`
- `upAxis = Y`
- No import errors in the console
- Mesh captures canopy mass and trunk geometry

Alternatively, open in Blender 3.0+: `File > Import > Universal Scene Description (.usda)`.

---

## Expected Outputs

| File | Location | Description |
|------|----------|-------------|
| `frame_NNNNNN.jpg` | `data/input/custom/tree/images/` | JPEG frames at 0.5 fps |
| `transforms.json` | `data/input/custom/tree/` | DA3 camera poses + depth |
| `instance_mask/*.png` | `data/input/custom/tree/instance_mask/` | Per-frame binary masks |
| `surface_100_whole.ply` | HoloScene exps dir | Whole-scene Poisson mesh |
| `tree.usda` | HoloScene exps dir | Final USD ASCII asset |

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `EXTRACT_FPS` | 0.5 | Adjust upward if COLMAP registration is poor |
| COLMAP `--SequentialMatching.overlap` | 50 | Frames matched per direction |
| COLMAP `--Mapper.min_num_matches` | 10 | Lowered from default 15 |
| `SAM3_MIN_SCORE` | 0.10 | Minimum mask confidence |
| `SAM3_MIN_FRAME_DURATION` | 40 | Minimum frames object must appear |
| HoloScene iterations | 256 | Gaussian splatting training steps |

---

## Hardware Requirements Summary

| Stage | Hardware | Approx. Time |
|-------|----------|-------------|
| FFmpeg + COLMAP | NVIDIA Quadro M3000M (local) | ~25 min |
| DA3 | NVIDIA H100 NVL (HPC) | ~45 min |
| SAM3 | NVIDIA L40S (HPC) | ~30 min |
| HoloScene | NVIDIA L40S (HPC) | ~3 hr |

---

## Known Challenges — Tree Dataset

The tree is the harder of the two Video2Sim objects. The following issues were encountered and addressed during the project:

**Sky background reduces SIFT coverage.** Regions of uniform blue sky above the canopy produce very few keypoints. COLMAP handles this by anchoring reconstruction on the bark texture of the trunk and the boundary between canopy and sky. Manual verification of the point cloud in the COLMAP GUI is essential before uploading to the HPC.

**Leaf boundary ambiguity in SAM3.** The outer edges of the spring canopy blend with the background lawn (similar green tones) in certain frames. SAM3's confidence score is lower at these edges. The minimum score threshold of 0.10 was chosen to retain coverage while filtering truly spurious detections.

**Wind-induced motion blur.** Leaf motion between frames introduces reconstruction artifacts in HoloScene, particularly at the canopy perimeter. The Poisson surface mesh captures the overall canopy mass but does not resolve individual leaves — this is an expected limitation of monocular reconstruction.

**USDA vs. USDC output.** The tree pipeline produces a `.usda` (USD ASCII) file. This is functionally identical to `.usdc` for import purposes but is larger on disk. It can be converted to binary `.usdc` using Isaac Sim's `File > Save As` after opening.

---

## Troubleshooting

**DA3 fails with Hugging Face 401 error:** Re-export `HF_TOKEN` in the current shell before submitting.

**HoloScene crashes with `AssertionError: mesh N does not exist`:** Apply the patch in `docs/holoscene_patch.md` — this is required for single-instance scenes.

**COLMAP registration rate < 80%:** Increase `EXTRACT_FPS` or inspect the video for dark/blurry sections and trim them before re-running.

**SAM3 masks cover the background buildings instead of the tree:** Make prompts more specific. Remove background descriptors (e.g., "Red brick buildings") and add foreground-only terms such as "deciduous tree trunk" and "tree canopy."

**HoloScene training stalls at Stage 0 (Marigold priors):** Verify `HF_TOKEN` is valid and that the Marigold model is accessible. Cache is stored at `~/video2sim-conda2/data/cache/da3`.

---

## Citation

If you use this pipeline in your work, please cite:

```bibtex
@misc{video2sim_tree_2026,
  title  = {Video2Sim: Tree Dataset Pipeline},
  author = {Rajashekhar, Chandana},
  year   = {2026},
  note   = {\url{https://github.com/Chandana9700/video2sim-tree}}
}
```

---

## License

This project is submitted as coursework for CSCI 5961 – AI Capstone – Fall 2026, Saint Louis University. Third-party tools (COLMAP, DA3, SAM3, HoloScene) retain their respective licenses.
