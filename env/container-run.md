# Running the ROCm container

All PyTorch work for `polaris` happens **inside** the
`robertrosenbusch/rocm6_gfx803_comfyui:5.7` container. ROCm 5.7 +
PyTorch 2.3.0a0 + HIP 5.7 + Python 3.10 (conda env at
`/opt/conda/envs/py_3.10/`). The host has no ROCm and no project-local
torch.

**Why 5.7 and not a newer ROCm:** path-validation surfaced an fp32 GEMM
correctness bug on gfx803 in every ROCm 6.x image we tested (6.1.2 and
6.4.3 give bit-identical broken output on `F.linear`). ROCm 5.7 is the
community-validated path where fp32 GEMM works correctly. See
`KNOWN_GOTCHAS.md` for the full verification matrix and the link to
`robertrosenbusch/gfx803_rocm` issue #55.

The image is the ComfyUI bundle (the only ROCm 5.7 prebuilt available
for gfx803). ComfyUI itself is dormant — we override the entrypoint with
`--entrypoint bash` so the container is a plain shell.

## Quick start

```
# from the project root:
./env/run-container.sh           # start (or attach to) container "polaris-dev"
```

The helper script:
- creates the container with the gfx803 flags on first invocation,
- starts it and `docker exec`s a shell on subsequent invocations,
- bind-mounts the project root at `/workspace`,
- overrides the ComfyUI entrypoint with bash,
- does **not** set `HIP_VISIBLE_DEVICES`, so both cards are visible
  inside.

To stop and remove the container: `./env/run-container.sh stop`.
To check its state: `./env/run-container.sh status`.

## First time inside the container

```
cd /workspace
python env/check-matmul.py --dtype fp32   # expect "ALL OK"
pip install -e .                          # installs deps + polaris.
```

`python` is the conda env's Python 3.10. The install persists as long as
the container exists (no `--rm`).

You'll see a harmless warning from pip about `scipy 1.8.1 requires
numpy<1.25.0` — scipy is preinstalled in the image and isn't used by
polaris; the warning has no effect.

## Manual `docker run` (equivalent to what the script does)

```
docker run -it \
  --name polaris-dev \
  --privileged \
  --network host \
  --ipc host \
  --shm-size 16G \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -e HSA_OVERRIDE_GFX_VERSION=8.0.3 \
  -e PYTORCH_ROCM_ARCH=gfx803 \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD:/workspace" \
  -w /workspace \
  --entrypoint bash \
  robertrosenbusch/rocm6_gfx803_comfyui:5.7
```

Notable flags and *why*:

- `--device /dev/kfd --device /dev/dri` — exposes **both** RX 480s.
  Passing a single `renderD12X` would hide the second card.
- `HSA_OVERRIDE_GFX_VERSION=8.0.3` + `PYTORCH_ROCM_ARCH=gfx803` —
  required for Polaris. The image already bakes these in, but explicit
  is safer if the image is updated.
- `--group-add video` — needed for `/dev/kfd` access.
- `--shm-size 16G` + `--ipc host` — required by PyTorch DataLoader workers.
- `--entrypoint bash` — bypass the default ComfyUI launch.
- **No `HIP_VISIBLE_DEVICES` at launch.** Setting it here masks a card.
  Set it *per-process inside the container* (see two-card pattern).

## Two-card pattern (one experiment per card)

In one host shell:
```
./env/run-container.sh           # shell A inside the container
HIP_VISIBLE_DEVICES=0 python -m polaris.train --config experiments/configs/dense.yaml
```

In a second host shell:
```
./env/run-container.sh           # exec's a second shell into the same container
HIP_VISIBLE_DEVICES=1 python -m polaris.train --config experiments/configs/moe.yaml
```

Two independent processes, one per GPU. No RCCL, no DDP (RCCL on gfx803
is unreliable). The sweep driver that schedules pairs across the cards
is `# TODO(human):` — yours to write.

## First-run sanity checks (inside the container)

```
rocminfo | grep -c 'Name: *gfx803'                  # expect 2
python -c "import torch; print(torch.cuda.device_count())"  # expect 2
python env/check-matmul.py --dtype fp32             # expect "ALL OK"
```

If `device_count()` is 1 instead of 2, check that you didn't set
`HIP_VISIBLE_DEVICES` at container launch and that both
`/dev/dri/renderD12*` nodes exist on the host. See
`~/ROCm-For-RX580/NOTES.md`.

If the matmul check fails, **stop** and read `KNOWN_GOTCHAS.md` — you
may have ended up on a ROCm 6.x image by accident.

## Gotchas

- **Host Python can't `import torch`.** Linters/IDEs on the host will
  complain about torch imports. Either ignore or install a CPU-only
  torch in a host venv *only for editor intellisense*, accepting that
  its version will differ from the container's. For this learning
  project, ignoring is fine.
- **Tests run inside the container.** `pytest` on the host can't import
  modules that import torch. Run `pytest` from a container shell.
- **The image is the ComfyUI bundle.** ~38 GB extracted. ComfyUI itself
  is dormant (we override the entrypoint). A clean 5.7 + PyTorch base
  doesn't exist as a prebuilt; building from
  `~/gfx803_rocm/rocm_5.7/Dockerfile_rocm57_pt23` takes hours.
- **File ownership.** Processes inside run as root by default; files
  written from inside land on the host root-owned. Fix with
  `--user $(id -u):$(id -g)` (may collide with /dev/kfd permissions —
  TODO(human): verify) or a periodic `sudo chown -R $USER:$USER .`.
- **Avoid host kernel 6.12 / 6.13.** Known segfaults with ROCm on
  gfx803.
- **Persistent vs ephemeral.** Script uses a persistent container
  (no `--rm`) so you can `docker exec` into it for the two-card pattern.
- **Don't accidentally launch ComfyUI.** If you `docker run` without
  `--entrypoint bash`, the container will try to start ComfyUI on
  port 8188. Harmless but confusing.

## TODOs left in this doc

- TODO(human): verify the exact tag on Docker Hub the day you pull;
  upstream may republish. Currently
  `robertrosenbusch/rocm6_gfx803_comfyui:5.7`.
- TODO(human): if you ever want a slimmer image, build from
  `~/gfx803_rocm/rocm_5.7/Dockerfile_rocm57_pt23` — clean PyTorch on
  ROCm 5.7 without the ComfyUI bundle. Hours of compile.
