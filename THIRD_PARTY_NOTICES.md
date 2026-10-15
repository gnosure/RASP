# Licensing and third-party notices

## Project additions and modifications

RASP additions and modifications are distributed under the **NVIDIA Source Code License for DiffPure**, reproduced without modification in the root [`LICENSE`](LICENSE). This follows the license already referenced by the bundled DiffPure classifier files. Third-party portions with their own license notices remain under those terms.

The root license limits use of the covered work and its derivatives to non-commercial research or evaluation, subject to its terms. This repository must not be described as wholly MIT-licensed or as granting unrestricted commercial use. This notice does not remove or replace any upstream copyright, attribution, or license conditions.

## Bundled source code

| Location | Identified source | License text / scope |
| --- | --- | --- |
| `classifiers/attribute_classifier.py`, `classifiers/attribute_net.py`, `classifiers/cifar10_resnet.py` | [NVIDIA DiffPure](https://github.com/NVlabs/DiffPure); existing file headers identify NVIDIA copyright (2022). | [Root LICENSE](LICENSE), the NVIDIA Source Code License for DiffPure. Retain the existing file headers and any embedded source attributions. |
| `guided_diffusion/` | [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion); file headers identify the corresponding upstream files. | [LICENSE_GUIDED_DIFFUSION](guided_diffusion/LICENSE_GUIDED_DIFFUSION), MIT, copyright OpenAI (2021), subject to individual file notices. |
| `score_sde/`, except files with a different notice | [Score-SDE PyTorch](https://github.com/yang-song/score_sde_pytorch); many files identify The Google Research Authors (2020). | [LICENSE_SCORE_SDE](score_sde/LICENSE_SCORE_SDE), Apache License 2.0. |
| `score_sde/op/fused_bias_act_kernel.cu`, `score_sde/op/upfirdn2d_kernel.cu` | [NVIDIA StyleGAN2](https://github.com/NVlabs/stylegan2); the kernel headers identify NVIDIA copyright (2019) and the Nvidia Source Code License-NC. | [LICENSE_STYLEGAN2](score_sde/op/LICENSE_STYLEGAN2). These individual notices take precedence over a blanket description of the parent directory as Apache-licensed. |

Some upstream files also acknowledge additional sources, such as OpenAI baselines, StyleGAN, and TRADES. Keep those embedded notices. This inventory records the sources identified by the bundled notices; it is not a reconstruction of every unmarked legacy file's history. An absent header does not establish that a file is an original RASP contribution.

Upstream license sources:

- [DiffPure license](https://github.com/NVlabs/DiffPure/blob/master/LICENSE)
- [guided-diffusion license](https://github.com/openai/guided-diffusion/blob/main/LICENSE)
- [Score-SDE license](https://github.com/yang-song/score_sde_pytorch/blob/main/LICENSE)
- [StyleGAN2 license](https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt)

## External dependencies and model resources

RobustBench, AutoAttack, PyTorch, torchvision, and the other packages listed in `requirements.txt` are installed separately. Their licenses continue to apply to those packages; the RASP root license does not relicense them.

Pretrained diffusion/classifier weights and datasets are also obtained separately from their providers. Links in `readme.md` do not grant additional rights to those resources. Do not bundle downloaded ImageNet data or model checkpoints with the source release without separately checking their distribution terms.
