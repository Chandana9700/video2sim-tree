# HoloScene Instance Mesh Fallback Patch

## Why This Patch Is Required

HoloScene's post-processing step (`instance_meshes_post_pruning`) asserts that a per-instance
mesh file exists after Gaussian splatting training. For single-instance or sparse scenes — such
as the tree dataset — the Poisson meshing step may not produce a file for every tracked
instance index, causing the following crash even after successful training:

```
AssertionError: mesh 1 does not exist
```

The patch replaces the hard assertion with a fallback that logs a warning and skips the missing
instance, allowing post-processing to complete and the `.usda` export to proceed normally.

---

## Location

```
src/holoscene/training/holoscene_train.py
```

Approximate line: **529** (inside the `instance_meshes_post_pruning` function).

---

## The Change

### Before (original line)

```python
assert os.path.exists(obj_i_mesh_path), f"mesh {obj_i} does not exist"
```

### After (replacement block)

```python
if not os.path.exists(obj_i_mesh_path):
    print(f"[WARNING] instance_meshes_post_pruning: mesh for instance "
          f"{obj_i} not found at {obj_i_mesh_path}. Skipping.")
    continue
```

---

## How to Apply

```bash
cd src/holoscene/training/

# 1. Back up the original
cp holoscene_train.py holoscene_train.py.bak

# 2. Open the file and locate the assertion (search for "does not exist")
#    Replace the assert line with the two-line fallback shown above.

# 3. Verify the edit
grep -n "does not exist\|WARNING.*mesh" holoscene_train.py
```

Expected output after the patch:

```
529:    print(f"[WARNING] instance_meshes_post_pruning: mesh for instance "
```

---

## Effect on Output

- Training output (`surface_100_whole.ply`, Gaussian splat checkpoints) is **not affected**.
- Instances with a generated mesh are processed normally.
- Instances without a mesh are skipped with a printed warning — this is expected for
  single-object scenes where only one instance index is active.
- The final `.usda` / `.usd` export proceeds to completion.

---

## Verification

After applying the patch, re-run HoloScene. The job log should show lines similar to:

```
[WARNING] instance_meshes_post_pruning: mesh for instance 1 not found at .../mesh_1.obj. Skipping.
surface_100_whole.ply save to ./exps/holoscene_tree/.../plots
```

If you still see the `AssertionError`, confirm the backup and re-apply the patch — the edit
may not have saved correctly.
