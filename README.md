# Fine-Tuning LLMs — A Hands-On Notebook Course

A 26-notebook, runnable course on fine-tuning large language models: from first
principles (tokenizers, loss, optimizer state) through every major method
(LoRA, QLoRA, DPO, GRPO, and more) to the practical end of the pipeline —
preparing your own data, saving/deploying a model, and debugging when it breaks.

> **What this is:** an *educational* resource. The demos use tiny datasets and small
> `max_steps` so they run fast and show *mechanics*, not benchmark numbers. For
> production training you'd reach for a framework like Unsloth, TRL, LLaMA-Factory,
> or Axolotl — this course teaches the concepts those frameworks automate.

## Who it's for

Anyone who can write basic Python and wants to actually understand fine-tuning —
not just copy a script. Each method notebook is **self-contained**: open it, run
the Setup cell, run the rest.

## Curriculum

**Foundations**
| # | Notebook | Covers |
|---|---|---|
| 01 | [Foundations](01_foundations.ipynb) | tokenizers, chat templates, loss, optimizer state, hyperparameters, evaluation, precision, data formats |
| 02 | [Supervised vs. self-supervised](02_concept_supervised_vs_selfsupervised.ipynb) | the two training objectives, and which one fine-tuning uses |
| 02b | [Can I fine-tune this?](02b_feasibility_preflight.ipynb) | VRAM estimator, data/method fit, smoke test, pre-flight checklist |
| 02c | [Prepare your own dataset](02c_prepare_your_data.ipynb) | load (CSV/JSON/Hub), clean, split, format to chat template |

**Core methods**
| # | Notebook | Covers |
|---|---|---|
| 03 | [Full fine-tuning](03_full_finetuning.ipynb) | update every weight — the baseline |
| 04 | [LoRA](04_lora.ipynb) | low-rank adapters + a rank chooser |
| 05 | [QLoRA](05_qlora.ipynb) | 4-bit base + adapters for consumer GPUs |
| 06 | [The math behind LoRA & QLoRA](06_lora_qlora_math.ipynb) | why low rank works, NF4 quantization |
| 07 | [Unsloth](07_unsloth.ipynb) | the single-GPU speed/VRAM accelerator |
| 07b | [Continued pre-training](07b_continued_pretraining.ipynb) | self-supervised domain adaptation on raw text (DAPT) |

**Instruction tuning & alignment**
| # | Notebook | Covers |
|---|---|---|
| 08 | [Instruction tuning](08_instruction_tuning.ipynb) | response-only loss masking |
| 09 | [Prompt tuning](09_prompt_tuning.ipynb) | soft prompts / prefix tuning |
| 10 | [DPO](10_dpo.ipynb) | Direct Preference Optimization |
| 11 | [GRPO](11_grpo.ipynb) | reward-function RL (the DeepSeek-R1 method) + its classic PPO-RLHF predecessor |
| 12 | [More PEFT](12_more_peft.ipynb) | DoRA, IA³, BitFit, adapters, and LoRA variants (rsLoRA, PiSSA, LoftQ, AdaLoRA, VeRA) |
| 13 | [More alignment](13_more_alignment.ipynb) | ORPO, KTO, SimPO, reward modeling |
| 14 | [Bootstrapping](14_bootstrapping.ipynb) | rejection sampling, distillation |
| 14b | [Model merging](14b_model_merging.ipynb) | TIES, DARE, SLERP, model soups via mergekit |

**Reference & operations**
| # | Notebook | Covers |
|---|---|---|
| 15 | [Decision guide & use cases](15_decision_guide_and_use_cases.ipynb) | which method to pick, when *not* to fine-tune |
| 15b | [Production frameworks](15b_production_frameworks.ipynb) | TRL, Unsloth, Axolotl, LLaMA-Factory — concept + minimal code each |
| 16 | [Glossary & references](16_glossary_and_references.ipynb) | terms + every method's source paper (arXiv) |
| 17 | [Save, load & deploy](17_save_load_deploy.ipynb) | save adapters, reload, merge, export (Hub/GGUF/vLLM) |
| 18 | [Troubleshooting & FAQ](18_troubleshooting_faq.ipynb) | errors→fixes + beginner FAQ |
| 19 | [Evaluation](19_evaluation.ipynb) | perplexity, exact-match, LLM-as-judge, `lm-evaluation-harness` |
| 20 | [Capstone](20_capstone.ipynb) | end-to-end fine-tune: data → train → evaluate → align → deploy |
| 21 | [Batch inference](21_batch_inference.ipynb) | run a saved fine-tune over a new dataset (Hub / CSV / JSONL), save predictions |

## Quickstart

Fine-tuning needs a GPU. Two easy paths:

**Google Colab (free T4):** open any notebook from this repo with the URL pattern
below (replace `tasmia008/Complete_Guidelines_LLM_FineTuning`):

```
https://colab.research.google.com/github/tasmia008/Complete_Guidelines_LLM_FineTuning/blob/main/04_lora.ipynb
```

**Local / cloud GPU:**

```bash
git clone https://github.com/tasmia008/Complete_Guidelines_LLM_FineTuning.git
cd Complete_Guidelines_LLM_FineTuning
pip install -r requirements.txt
jupyter lab
```

> `bitsandbytes` (4-bit/QLoRA) and `unsloth` are **NVIDIA-CUDA only**. On CPU or
> Apple `mps` you can still run plain LoRA — skip the 4-bit cells.

## Suggested path

New to fine-tuning? Do **01 → 02 → 02b → 02c**, then **04 (LoRA)** and **06 (math)**.
Add **05 (QLoRA)** and **17 (deploy)** once you want to ship something. Reach for the
rest when a specific need comes up.

## Notes

- Each notebook prints its installed library versions (an env-check cell) for reproducibility.
- APIs verified against current `transformers` / `trl` / `peft` / `unsloth` releases.
- Every method links to its original paper (arXiv IDs in notebook 16).

## License

Released under the [MIT License](LICENSE).
# Complete_Guidelines_LLM_FineTuning
