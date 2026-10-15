# RASP

Diffusion-based adversarial purification with a timestep-dependent, sample-adaptive low-frequency radius and optional frequency guidance. This repository provides evaluation code for CIFAR-10 and ImageNet.

The main evaluation entry point is [`test_new.py`](test_new.py). It imports [`pure_new_adaptive_fixed.py`](pure_new_adaptive_fixed.py), which inherits the purification pipeline from [`pure_new.py`](pure_new.py) and overrides the adaptive-radius calculation. Both files are required.

## Repository structure

```text
RASP/
|-- attacks/                      # PGD-EOT, BPDA, and AutoAttack wrappers
|-- diffusion_configs/            # Backbone configurations
|-- guided_diffusion/             # ImageNet diffusion backbone implementation
|-- score_sde/                    # CIFAR-10 diffusion backbone implementation
|-- scripts/                      # Evaluation commands
|-- svhn/                         # Imported by the shared model loader
|-- load_data.py
|-- load_model.py
|-- path.py
|-- pure_new.py
|-- pure_new_adaptive_fixed.py
|-- requirements.txt
|-- test_new.py
`-- utils.py
```

`ddp_test.py` is a legacy entry point. It requires `purification.py`, which is not included in this release. Use `test_new.py` and the scripts below for the current method.

## Installation

Evaluation requires Linux, Bash, an NVIDIA GPU, and a CUDA-enabled PyTorch installation. The scripts use the Python executable from the currently activated environment. They default to two GPUs, but also support one GPU.

The following is a Python 3.9 installation template for the legacy evaluation code, not a locked or GPU-validated reproduction environment. `requirements.txt` pins PyTorch 2.0.1, torchvision 0.15.2, NumPy 1.26.4, and PyArrow 11.0.0. Other package versions and the RobustBench source are not fully locked. For exact reproduction, retain the package versions, source revisions, and GPU configuration of the environment used for the reported experiments.

```bash
conda create -n rasp python=3.9 -y
conda activate rasp

# CUDA 11.8 example; choose a build supported by your GPU and driver.
python -m pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r requirements.txt

python -m pip check
```

The PyTorch pairing follows the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/). AutoAttack is installed from its [official repository](https://github.com/fra31/auto-attack) at revision `a39220048b3c9f2cca9a4d3a54604793c68eca7e`, matching the dependency declared by the inspected [RobustBench source](https://github.com/RobustBench/robustbench/blob/master/setup.py). It is imported at startup, including when running PGD. NumPy 1.26.4 avoids the NumPy 2 incompatibility of the legacy PyArrow/PyTorch stack. A matching CUDA toolkit and compiler may also be needed if the bundled extensions are compiled locally.

`xformers` and `nested_dict` are not needed by the current evaluation entry point and are omitted. The bundled upstream guided-diffusion training/data utilities also import `blobfile` and `mpi4py`; these are outside the evaluation dependency list. `torchrun` evaluation uses PyTorch distributed directly and does not require MPI.

Before launching a long experiment, check imports, NumPy interoperability, and GPU visibility:

```bash
python -c "import sys, numpy as np, torch, torchvision, pyarrow, robustbench, autoattack; print(sys.executable); print('numpy:', np.__version__); print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPUs:', torch.cuda.device_count()); print(torch.from_numpy(np.zeros(3, dtype=np.float32)))"
```

## Pretrained models

Download weights separately and arrange them as follows:

```text
RASP/
|-- pretrained/
|   `-- guided_diffusion/
|       |-- checkpoint_8.pth
|       `-- 256x256_diffusion_uncond.pt
`-- models/
    `-- cifar10/
        |-- Linf/
        |   `-- weights-best.pt
        `-- L2/
            `-- Standard.pt
```

| Dataset / component | Model | Weight source / loading behavior |
| --- | --- | --- |
| CIFAR-10 diffusion | NCSN++ with the VP-SDE configuration; EMA weights | Obtain the `vp/cifar10_ddpmpp_deep_continuous` checkpoint listed by [DiffPure](https://github.com/NVlabs/DiffPure#data-and-pre-trained-models) and save it as `pretrained/guided_diffusion/checkpoint_8.pth`. |
| ImageNet diffusion | Unconditional 256x256 guided-diffusion model | Download [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) from [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion). |
| CIFAR-10 WRN-28-10 | RobustBench `Standard` model | Automatically loaded/downloaded through `robustbench.utils.load_model(model_name='Standard', dataset='cifar10', threat_model='L2')`. |
| CIFAR-10 WRN-70-16 | `DMWideResNet`, depth 70, width 16, Swish | Follow the WRN-70-16 checkpoint instructions in [DiffPure](https://github.com/NVlabs/DiffPure#data-and-pre-trained-models) and place the matching checkpoint at `models/cifar10/Linf/weights-best.pt`. |
| ImageNet classifier | torchvision ResNet-50, `IMAGENET1K_V2` | Automatically loaded/downloaded by torchvision. |

Diffusion checkpoint paths are configured in `path.py`. The WRN-70-16 path is configured separately in `utils.py`; its loader expects a checkpoint with a `model_state_dict` entry. The CIFAR-10 diffusion loader restores model, EMA, optimizer, and step entries rather than a bare model state dictionary.

CIFAR-10 uses the Score-SDE NCSN++ backbone, not the NVIDIA EDM interface. For ImageNet, the first three diffusion output channels are used as the noise prediction. `--clf_arch` selects the CIFAR-10 classifier only; ImageNet always loads ResNet-50, even if the printed argument namespace contains the default WRN name.

## Datasets

### CIFAR-10

The test set is downloaded automatically to `./dataset/cifar10/` relative to the repository root.

### ImageNet

Prepare ImageNet-1K locally in the layout expected by `torchvision.datasets.ImageNet`. This release uses image files, not the optional LMDB loader.

For example:

```text
/data/datasets/
`-- imagenet/
    |-- meta.bin
    `-- val/
        |-- n01440764/
        |   `-- *.JPEG
        `-- ...
```

If `meta.bin` has not been generated, torchvision also needs the ImageNet devkit archive to prepare the metadata. Follow the dataset preparation requirements of your installed torchvision version.

Set the **parent of the `imagenet` directory** in `path.py`:

```python
imagenet_path = '/data/datasets'
```

The loader appends `/imagenet` to this value. Do not set it to `/data/datasets/imagenet/val`.

The current ImageNet preprocessing resizes the shorter edge to 232, center-crops to 224x224, and converts images to tensors in `[0, 1]`. The purification model resizes its input to 256x256; the classifier path resizes it back to 224x224 and applies ImageNet normalization. Use identical input preprocessing when comparing methods, and regenerate adversarial examples after changing preprocessing.

### Evaluation subset

`test_new.py` evaluates a subset of **512 images**. The subset is chosen with `numpy.random.RandomState(0)` in `load_data.py`; `--seed` controls experiment randomness but does not change this subset seed. There is currently no `--num_samples` argument. Change the subset size in the test entry point if needed.

For distributed runs, choose a world size that divides 512 to avoid sampler padding and duplicate evaluation samples. One or two GPUs satisfy this condition.

## Evaluation

Run commands from the repository root after activating your environment. The scripts also resolve the repository root themselves, so their relative model and data paths do not depend on the launch directory.

| Script | Dataset | Classifier | Attack | Iterations | EOT samples | Ensemble runs |
| --- | --- | --- | --- | --- | --- | --- |
| `scripts/cifar10.sh` | CIFAR-10 | WRN-28-10 | PGD-Linf | 200 | 20 | 5 |
| `scripts/cifar10 _v2.sh` | CIFAR-10 | WRN-70-16 | PGD-Linf | 200 | 20 | 5 |
| `scripts/bpda.sh` | CIFAR-10 | WRN-28-10 | BPDA-Linf | 200 | 20 | 5 |
| `scripts/imagenet.sh` | ImageNet | ResNet-50 | PGD-Linf | 20 | 20 | 5 |

Run one experiment at a time on the selected GPUs:

```bash
# CIFAR-10, WRN-28-10.
CUDA_VISIBLE_DEVICES=0,1 bash scripts/cifar10.sh

# CIFAR-10, WRN-70-16. Preserve the space in this filename.
CUDA_VISIBLE_DEVICES=0,1 bash "scripts/cifar10 _v2.sh"

# CIFAR-10, BPDA.
CUDA_VISIBLE_DEVICES=0,1 bash scripts/bpda.sh

# ImageNet.
CUDA_VISIBLE_DEVICES=0,1 bash scripts/imagenet.sh
```

By default, `CUDA_VISIBLE_DEVICES=0,1`. The number of worker processes follows the number of comma-separated GPU IDs. Each GPU runs its own model replica on a different subset of the data. GPU IDs are remapped locally, so physical GPUs `2,3` become `cuda:0,1` inside the workers.

Override GPUs or the rendezvous port as needed:

```bash
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29566 bash scripts/imagenet.sh

# Single-GPU example.
CUDA_VISIBLE_DEVICES=0 bash scripts/cifar10.sh
```

Additional command-line arguments are appended to `test_new.py`, allowing scalar settings to be overridden:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/cifar10.sh --seed 2 --n_iter 50
CUDA_VISIBLE_DEVICES=0,1 bash scripts/bpda.sh --clf_arch wideresnet-70-16
```

Both frequency guidance and adaptive radius are enabled in the scripts. To disable a `store_true` flag for an ablation, remove the corresponding `--use_freq_guidance` or `--use_adaptive_radius` line from a separate copy of the script; there are no inverse flags.

### Purification settings in the scripts

| Setting | CIFAR-10 | ImageNet |
| --- | --- | --- |
| Defense maximum timesteps | `1000` | `50,50,50,50,50,50,50,50` |
| Defense denoising steps | `100` | `5,5,5,5,5,5,5,5` |
| Attack maximum timesteps / denoising steps | `1000 / 1` | `50 / 1` |
| Sampling method | `ddpm` | `ddpm` |
| Reference forward-noise step | 50 | 50 |
| Radius min / max / gamma | `2 / 14 / 1.0` | `6 / 16 / 1.0` |
| Adaptive mix / threshold / temperature | `0.2 / 0.18 / 0.10` | `0.2 / 0.18 / 0.10` |
| Amplitude / phase weights | `1.0 / 0.25` | `1.0 / 0.25` |
| Guidance scale / start ratio | `3e-4 / 0.6` | `5e-4 / 0.6` |
| Seed | 1 | 1 |

These are the supplied script configurations, not a claim that they are optimal for every backbone or attack. Timesteps specified in the CLI are converted to zero-based indices: a maximum value of `50` starts at timestep 49. In the corrected adaptive controller, a larger internal spectral discrepancy gives a larger spectral radius at a fixed base radius; this does not imply that every adversarial input must have a larger radius than its clean counterpart.

### Attack settings and scope

Inputs and perturbation budgets are expressed in the `[0, 1]` image domain. The following budgets are currently hardcoded in `test_new.py`:

| Dataset | `--attack_method` | Norm | Epsilon |
| --- | --- | --- | --- |
| CIFAR-10 | `pgd`, `bpda`, `aa` | Linf | `8/255` |
| CIFAR-10 | `pgd_l2`, `aa_l2` | L2 | `0.5` |
| ImageNet | `pgd`, `aa` | Linf | `4/255` |
| ImageNet | `aa_l2` | L2 | `4/255` (current implementation) |

ImageNet `pgd_l2` and `bpda` are not implemented in the dataset-specific attack branch. There is no CLI epsilon option. The ImageNet L2 entry above records the actual code; it must not be reported as an L2 budget of `3.0` unless the code and corresponding experiments are changed.

The PGD wrappers use `--n_iter`, `--eot`, and a step size of `0.007`. Attacks are generated against `attack_forward`, which uses the shorter attack denoising schedule and has frequency guidance disabled. Reported defense accuracy is then measured using `defense_forward` and majority-vote ensembling. This is a surrogate attack configuration, not an end-to-end attack through the full defense schedule and ensemble.

The AutoAttack wrappers have the following behavior:

- `aa`: custom AutoAttack with APGD-CE and APGD-DLR. It does not include FAB or Square, and it does not configure the randomized-defense EOT setting itself.
- `aa_l2`: AutoAttack `version='rand'`, whose settings are supplied by the installed AutoAttack implementation.
- Both wrappers accept `attack_steps` and `eot` in their constructors but do not forward those values into AutoAttack. Changing CLI `--n_iter` or `--eot` therefore does not configure the internal AutoAttack iterations or EOT.

For example, after installing AutoAttack:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/cifar10.sh --attack_method aa
CUDA_VISIBLE_DEVICES=0,1 bash scripts/cifar10.sh --attack_method aa_l2
```

Describe the actual AutoAttack configuration and installed version when reporting results; do not label the custom Linf subset as the full standard AutoAttack suite.

## Background execution with tmux

Activate the environment, enter the repository root, then start a detached session. The activated environment's executable directory is passed explicitly, and the current directory is passed to tmux:

```bash
conda activate rasp
tmux new-session -d -s rasp_imagenet -c "$PWD" \
  /bin/bash -c 'export PATH="$1:$PATH"; CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=29566 bash scripts/imagenet.sh > imagenet_pgd.log 2>&1' \
  _ "$(dirname "$(command -v python)")"
```

For an existing environment such as `freq_env`, activate that environment instead of `rasp`. Replace the script and log name for a CIFAR-10 run. Session names must be unique.

Check progress:

```bash
tmux ls
tail -f imagenet_pgd.log
tmux attach -t rasp_imagenet
```

Detach from an attached session with `Ctrl+b`, then `d`. Closing the local editor or shutting down the local computer does not normally stop a detached tmux session on the remote server. Server shutdowns, server-side session policies, and Python/CUDA errors can still terminate the job.

To stop this experiment, terminate only its session:

```bash
tmux kill-session -t '=rasp_imagenet'
```

## Outputs and interpretation

The final aggregated line reports `num_samples`, `acc_nat`, and `acc_adv`:

- `acc_nat`: accuracy after purifying clean inputs, not the raw classifier's clean accuracy.
- `acc_adv`: accuracy after purifying the generated adversarial inputs.
- `num_ensemble_runs`: the number of independently purified predictions combined by majority vote.

Intermediate progress lines are local to each rank. The final totals are summed across ranks; each rank may print the same aggregated result. Do not add these final lines together.

Console output is also appended to `output_logs.txt`. The tmux example redirects stdout and stderr to a dedicated experiment log. `--exp` is a label in the printed arguments; it does not create a separate output directory.

The evaluator also writes `original/`, `adv/`, and `pure_images/`. In the current implementation, image filenames do not include the rank and can overwrite each other in multi-GPU runs; directory creation can also race. ImageNet purified tensors returned by the classifier path are normalized, so the saved purified images are not a reliable visualization export. Use the numeric logs for evaluation, and address image export separately before using these files in figures. Separate experiments launched in the same checkout also share output locations.

## Troubleshooting

- **NumPy/PyArrow import failures:** check the active Python path, NumPy version, and `torch.from_numpy` using the installation smoke check. The legacy stack should use NumPy 1.x. Avoid repeatedly mixing Conda and pip reinstalls in an already inconsistent environment.
- **`No module named autoattack`:** install the official AutoAttack package listed above, even for a PGD run.
- **Checkpoint not found:** inspect `path.py` and the separate WRN-70-16 path in `utils.py`.
- **ImageNet directory not found:** `imagenet_path` must point to the parent of `imagenet/`.
- **CUDA out of memory:** reduce `--batch_size`; it must satisfy the current 512-sample assertion. Do not change attack budgets or iterations merely to hide memory errors.
- **Port already in use:** supply a different `MASTER_PORT`.
- **`ChildFailedError`:** inspect the preceding rank traceback in the experiment log. This launcher summary does not identify the underlying Python error by itself.

## Acknowledgments

The implementation uses code or model resources from [Score-SDE](https://github.com/yang-song/score_sde_pytorch), [guided-diffusion](https://github.com/openai/guided-diffusion), [DiffPure](https://github.com/NVlabs/DiffPure), [RobustBench](https://github.com/RobustBench/robustbench), [AutoAttack](https://github.com/fra31/auto-attack), and [torchvision](https://github.com/pytorch/vision).

## License

RASP additions and modifications use the [NVIDIA Source Code License for DiffPure](LICENSE), which limits the covered work and derivatives to non-commercial research or evaluation, subject to its terms. The bundled Score-SDE, guided-diffusion, and StyleGAN2 components retain their individual licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the source and license inventory; this repository is not uniformly MIT-licensed.

Keep the existing copyright and attribution notices, `score_sde/LICENSE_SCORE_SDE`, `guided_diffusion/LICENSE_GUIDED_DIFFUSION`, and `score_sde/op/LICENSE_STYLEGAN2` when distributing the corresponding code. Pretrained weights and datasets are obtained separately from their original providers.

The root `.gitignore` excludes datasets, checkpoints, caches, logs, and generated image/statistics directories. It does not delete those files or remove files that Git already tracks. Keep selected paper figures or curated result tables in a separate tracked directory when needed.
