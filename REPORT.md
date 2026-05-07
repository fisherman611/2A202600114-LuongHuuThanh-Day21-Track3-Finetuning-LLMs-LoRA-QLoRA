# Lab 21 Report - LoRA Rank Experiment

**Student:** Luong Huu Thanh  
**Model:** Qwen2.5-3B-bnb-4bit (Unsloth)  
**Dataset:** `medalpaca/medical_meadow_medqa` (300 samples, Alpaca format)

---

## 1. Setup

- **Base model:** `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset size:** 300 MedQA samples
- **Max sequence length:** 512 (p95 = 405, rounded to power of 2)
- **GPU:** Tesla T4 (16 GB VRAM)
- **Estimated training cost:** ~$0.14 (~23.5 minutes at $0.35/hour)
- **Target modules:** all linear layers (`q`, `k`, `v`, `o`, `gate`, `up`, `down`)
- **Hugging Face model:** <https://huggingface.co/fisherman611/recipe-qwen2.5-3b-lora>

---

## 2. Rank Experiment Results

Source: `adapters/results/rank_experiment_summary.csv`

| Rank | Alpha | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|---|---:|---:|---:|---:|---:|---:|
| 8  | 16  | 14,966,784  | 6.40 | 7.71 | 1.2258 | 3.4068 |
| 16 | 32  | 29,933,568  | 9.48 | 5.07 | 1.2516 | 3.4961 |
| 64 | 128 | 119,734,272 | 7.61 | 9.48 | 1.3906 | 4.0171 |

Reference baseline from notebook logs:
- **Base model eval loss:** 1.9042
- **Base model perplexity:** 6.71

---

## 3. Analysis

- **Best quality:** `r=8` gives the lowest eval loss (1.2258) and perplexity (3.4068).
- **Diminishing returns:** increasing rank from 8 to 16 and 64 does not improve evaluation quality on this 300-sample dataset.
- **Potential over-parameterization:** `r=64` has 8x more trainable parameters than `r=8` but worse perplexity.
- **Resource trade-off:** `r=8` is also faster than `r=16`, while `r=64` uses the most VRAM.

Takeaway: on small, domain-specific instruction datasets, lower-rank adapters can provide better generalization and better cost-efficiency than high-rank adapters.

---

## 4. Qualitative Comparison (5 MedQA Prompts)

Source: `adapters/results/qualitative_comparison.csv`

| # | Prompt (shortened) | Base model output | Fine-tuned output (`r=16`) |
|---|---|---|---|
| 1 | 45M, sudden severe chest pain radiating to back | Listed multiple options + final `A: Acute aortic dissection` | `A: Acute aortic dissection` |
| 2 | 32F, butterfly rash, anti-dsDNA positive | Listed treatment options only | `B: Methotrexate` |
| 3 | 65M smoker, central lung mass | `C: Small cell carcinoma` | `A: Squamous cell carcinoma` |
| 4 | 5-year-old, fever + barking cough + stridor | `A: Bacterium` | `D. Adenovirus` |
| 5 | 55-year-old cirrhosis + hematemesis | `B: Variceal bleeding` | `A: Nonvaricosal esophageal varices` |

Observations:
- The fine-tuned model is generally more concise and decisive.
- Some outputs are better formatted after tuning, but medical correctness should still be validated by domain evaluation metrics and expert review.

---

## 5. Conclusion

- For this experiment, **LoRA rank `r=8` is the best ROI point**.
- Increasing rank beyond 8 did not improve loss/perplexity and increased compute cost.
- Recommended practical range for similar small medical QA tasks: **`r=8` to `r=16`**, with `r=8` as the default starting point.

---

## 6. What I Learned

- Rank should be tuned empirically; larger rank is not automatically better.
- With small datasets, adapter capacity can exceed signal and hurt generalization.
- Basic data/length profiling (like p95-based sequence length) helps keep VRAM usage efficient without truncating too aggressively.
