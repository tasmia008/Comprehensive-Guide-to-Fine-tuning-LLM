# Fine-tuning — a notebook course

The consolidated reference, split into **26 standalone notebooks**, one per concept.
Each method notebook carries its own **Setup** cell, so you can open any single one and run it.

## Suggested order

- **`01_foundations.ipynb`** — Foundations — tokenizers, templates, loss, optimizer, hyperparams, eval, precision, data
- **`02_concept_supervised_vs_selfsupervised.ipynb`** — Concept — supervised vs. self-supervised fine-tuning
- **`02b_feasibility_preflight.ipynb`** — Can I fine-tune this? — feasibility & pre-flight
- **`02c_prepare_your_data.ipynb`** — Prepare your own dataset — load, clean, split, format
- **`03_full_finetuning.ipynb`** — Full fine-tuning
- **`04_lora.ipynb`** — LoRA
- **`05_qlora.ipynb`** — QLoRA
- **`06_lora_qlora_math.ipynb`** — The math behind LoRA & QLoRA
- **`07_unsloth.ipynb`** — Unsloth (accelerator)
- **`07b_continued_pretraining.ipynb`** — Continued pre-training — domain adaptation (self-supervised)
- **`08_instruction_tuning.ipynb`** — Instruction tuning
- **`09_prompt_tuning.ipynb`** — Prompt tuning (soft prompts)
- **`10_dpo.ipynb`** — DPO — Direct Preference Optimization
- **`11_grpo.ipynb`** — GRPO — reward-function RL
- **`12_more_peft.ipynb`** — More PEFT — DoRA, IA³, BitFit, adapters
- **`13_more_alignment.ipynb`** — More alignment — ORPO, KTO, reward modeling
- **`14_bootstrapping.ipynb`** — Bootstrapping — rejection sampling, distillation
- **`14b_model_merging.ipynb`** — Model merging — TIES, DARE, SLERP, model soups (mergekit)
- **`15_decision_guide_and_use_cases.ipynb`** — Decision guide & use cases
- **`15b_production_frameworks.ipynb`** — Production frameworks — TRL, Unsloth, Axolotl, LLaMA-Factory
- **`16_glossary_and_references.ipynb`** — Glossary & references
- **`17_save_load_deploy.ipynb`** — Save, load & run inference (deploy)
- **`18_troubleshooting_faq.ipynb`** — Troubleshooting & FAQ
- **`19_evaluation.ipynb`** — Evaluation — perplexity, task metrics, LLM-as-judge, benchmarks
- **`20_capstone.ipynb`** — Capstone — an end-to-end fine-tune: data → train → evaluate → deploy
- **`21_batch_inference.ipynb`** — Batch inference — run a saved fine-tune over a new dataset (HF / CSV / JSONL)

## How to use
- **New to fine-tuning?** Do `01_foundations` then `02_concept`, then `04_lora` and
  `06_lora_qlora_math`. Skip the rest until you need them.
- Each notebook prints its installed library versions (env-check) for reproducibility.
- Demo datasets are tiny and `max_steps` are small — outputs show *mechanics*, not benchmark
  numbers. Swap in real data/schedules for real results.
- **Hardware:** full FT is memory-heavy; QLoRA / Unsloth / bitsandbytes are NVIDIA-only;
  LoRA, prompt tuning, IA³, BitFit run (slowly) on CPU/`mps`.

## Verify any claim
Every method maps to a source paper in `16_glossary_and_references.ipynb` (arXiv IDs), and the
official docs (PEFT / TRL / Unsloth / bitsandbytes / adapters) are the live API source of truth.
