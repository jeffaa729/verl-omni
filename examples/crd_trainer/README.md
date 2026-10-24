# CRD Trainer

Last updated: 10/24/2026.

This example post-trains `stabilityai/stable-diffusion-3.5-medium` with
[CRD](../../docs/algo/crd.md) on OCR, using LoRA, the V1 synchronous trainer,
`vllm-omni` rollout and `Qwen/Qwen2.5-VL-3B-Instruct` as the reward model.
The algorithm document describes the loss, policy states and differences from
its paper and reference implementation.

## Installation

Follow the [installation guide](../../docs/start/install.md) and accept the
SD3.5-Medium access terms on Hugging Face. Install OCR dependencies:

```bash
uv pip install -e ".[ocr]"
```

The recipe defaults to two GPUs: one actor/rollout GPU and one OCR reward GPU.

## Prepare the dataset

Get `train.txt` and `test.txt` from the
[Flow-GRPO OCR dataset](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr),
place them under `$OCR_WORKSPACE/data/ocr`, and convert them:

```bash
export OCR_WORKSPACE=/path/to/workspace
mkdir -p "$OCR_WORKSPACE/data/ocr/sd3"
python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
  --input_dir "$OCR_WORKSPACE/data/ocr" \
  --output_dir "$OCR_WORKSPACE/data/ocr/sd3"
```

## Run

Run from the repository root:

```bash
bash examples/crd_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh \
  data.val_max_samples=128 data.validation_shuffle=False data.val_batch_size=1
```

Defaults are 100 iterations at 512×512, four prompts per actor GPU and six images
per prompt, with validation before training and every 10 iterations. One actor
GPU performs four optimizer updates per iteration. Keep evaluation prompts,
seed and generation settings fixed across checkpoints. Validation uses the `old`
sampling EMA adapter.

Use `NUM_GPUS_ACTOR_ROLLOUT`, `NUM_GPUS_REWARD`, `GROUP_SIZE`, `IMAGE_RESOLUTION`,
`TOTAL_TRAINING_STEPS` and `OUTPUT_DIR` to override the defaults. Hydra overrides
are accepted last. Whole prompt groups must fit in each actor micro batch.
The recipe starts fresh and retains the latest two checkpoints; use a new
`OUTPUT_DIR` for each experiment. Resume with `trainer.resume_mode=resume_path` and
`trainer.resume_from_path=/path/to/global_step_N`.

For a DGPO comparison with the same prompts, reward, initialization and training
budget, use a separate output directory:

```bash
OUTPUT_DIR=checkpoints/dgpo/crd_comparison \
  bash examples/crd_trainer/sd35/run_dgpo_comparison.sh \
  data.val_max_samples=128 data.validation_shuffle=False data.val_batch_size=1
```

Both recipes sample at CFG 1 with 10 steps and evaluate at CFG 4.5 with 40 steps.
This comparison matches the CRD experiment settings rather than DGPO's default
recipe. Training and validation rewards use different prompts and generation settings.

## Tested configuration

- Two NVIDIA A100-SXM4-80GB GPUs, with one actor/rollout GPU and one reward GPU.
- Python 3.12, PyTorch `2.13.0+cu130`, vLLM `0.30.0`, repository-pinned
  vLLM-Omni and verl, diffusers `0.40.0`, native/SDPA attention.
- Tiny-model training and distributed FSDP2 checkpoint restoration passed.
  Real SD3.5-Medium OCR training passed fresh iterations 1–2 and resumed
  iterations 3–4 at 256×256 with two images per prompt.

## GPU validation

Both runs completed 100 iterations (400 optimizer updates) at 512×512 with four
prompts and six images per prompt. Evaluation used 128 fixed held-out prompts,
one image per prompt, seed 42, and 11 passes at iterations 0, 10, …, 100.

| Metric | CRD | DGPO |
| --- | --- | --- |
| Initial OCR reward | 0.7671 | 0.7671 |
| Final OCR reward | 0.8312 | 0.8208 |
| Best observed reward | 0.8754 (iteration 30) | 0.8400 (iteration 90) |
| Mean training iteration, excluding evaluation | 51.6 s | 35.3 s |
| Training images per second, excluding evaluation | 0.465 | 0.679 |
| Sampled actor/rollout GPU memory peak | 28.1 GiB | 50.3 GiB |
| Sampled reward GPU memory peak | 57.1 GiB | 57.1 GiB |

The initial baseline is pretrained SD3.5-Medium before RL updates; images and
scores matched exactly between runs. Memory values are whole-device samples
from `nvidia-smi`. Both runs completed without nonfinite-gradient skips.

This is a single-seed comparison. The final CRD-minus-DGPO reward difference was
0.0104, with a paired prompt-bootstrap 95% interval of [−0.0489, 0.0723]; it does
not establish superiority. Both baseline-gain intervals also include zero.
CRD's final reward was below its iteration-30 peak. Only checkpoints 90 and 100
were retained; best observed scores refer to saved evaluations. OCR reward is
transcription agreement, not a measure of overall image quality.
