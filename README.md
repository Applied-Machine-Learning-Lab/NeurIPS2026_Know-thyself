
<div align="center">

<img src="assets/title.png" alt="Know-thyself: Feeling of Knowing in Large Language Models" width="80%">
<hr>

Zichuan Fu<sup>1</sup>, Xian Wu<sup>2</sup>, Jingtong Gao<sup>1</sup>, Wenlin Zhang<sup>1</sup>, Jiaxuan Li<sup>1</sup>, Binhao Wang<sup>1</sup>,<br>
Yimin Deng<sup>1,3</sup>, Guojing Li<sup>1,4</sup>, Xiaopeng Li<sup>1</sup>, Derong Xu<sup>1,5</sup>, Yefeng Zheng<sup>6</sup>, Xiangyu Zhao<sup>1</sup>

<sup>1</sup>City University of Hong Kong &nbsp; <sup>2</sup>Tencent Jarvis Lab &nbsp; <sup>3</sup>Xi'an Jiaotong University<br>
<sup>4</sup>Renmin University of China &nbsp; <sup>5</sup>University of Science and Technology of China &nbsp; <sup>6</sup>Westlake University

<h4>
<a href="https://fzkuji2026.github.io/Know-thyself/">🌐 Project Page</a>
</h4>

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce)](https://neurips.cc/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Post-training updates change what an LLM can answer, but its judgment of whether it can answer, its functional **feeling of knowing (FoK)**, may not follow. We train two channels: a knowledge channel that changes answerability and a FoK channel that judges the current knowledge state, in three stages:

- **OP-SFT** (on-policy SFT) updates knowledge from corrected on-policy answers.
- **KΔ-FT** (knowledge-delta FoK training) adapts the FoK judgment across intermediate knowledge states.
- **FGPO** (FoK-guided policy optimization) trains the answer or explanation that follows the judgment.

## 🏆 Results

One TriviaQA knowledge update on Qwen2.5-7B-Instruct (Table 2 of the paper). Bias is the signed mean of illusion and omission errors, closer to 0 is better.

<div align="center">

| | Train Acc. | Train AUROC | Train Bias | Test Acc. | Test AUROC | Test Bias | MMLU | ARC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.680 | 0.702 | 0.381 | 0.672 | 0.688 | 0.402 | 0.687 | 0.811 |
| Base + EliCal | 0.637 | 0.808 | 0.263 | 0.622 | **0.810** | **0.254** | 0.709 | 0.504 |
| Vanilla K-SFT | 0.874 | 0.458 | 0.500 | 0.546 | 0.519 | 0.500 | 0.713 | **0.840** |
| Vanilla K-SFT + Prob | 0.874 | 0.590 | 0.495 | 0.546 | 0.541 | 0.498 | 0.713 | 0.840 |
| **Ours** | **0.961** | **0.817** | **0.245** | **0.692** | 0.809 | 0.308 | 0.711 | 0.830 |

</div>

## 🚀 Quick start

```bash
git clone https://github.com/fzkuji2026/Know-thyself.git && cd Know-thyself
bash setup_deps.sh            # LLaMA-Factory, EasyR1, lm-eval-harness and EliCal at our commits, with our patches
pip install -e third_party/LlamaFactory -e third_party/EasyR1
```

Train the three stages on Qwen2.5-7B-Instruct (8 GPUs):

```bash
bash scripts/stage1_op_sft.sh    # OP-SFT: knowledge LoRA on corrected answer spans
bash scripts/stage2_kdelta.sh    # KΔ-FT: FoK LoRA over the OP-SFT checkpoint trajectory
bash scripts/stage3_fgpo.sh      # FGPO: GRPO with the FoK-guided reward
```

The scripts read model, Python and data locations from environment variables (`BASE_MODEL`, `MODEL`, `PY`, `TORCHRUN`, `OPSFT_SOURCE_DIR`, `AUX_ROOT`, `INIT_MODEL`, `QUERIES`, `EASYR1_ROOT`) and write to `runs/` and `results/`.

- Stage 1 expects sampled and corrected on-policy answers from `src/data/op_ssft_data_gen.py`.
- Stage 2 trains on answers generated under every saved OP-SFT checkpoint (`scripts/eval_stage1.sh`).
- Stage 3 starts from the merged Stage 2 model (`INIT_MODEL`). It uses the FGPO reward `easyr1_patches/reward_function/judgment_answer_think.py` with the `<think>`/`<answer>`/`<response>` prompt, and needs an answerability label per training question (`ability`, `label` or `is_correct`).

Baselines (Verbal-0, Verbal-10, Prob, N-Prob, Consis-Lex, Consis-Sem, SemEnt and EliCal):

```bash
python -m src.baselines.run_baselines --model Qwen/Qwen2.5-7B-Instruct \
    --baselines verbalized verbal_10 prob n_prob consis_lex consis_sem semantic_entropy elical \
    --output_dir results/baselines
```

Answer correctness with the Qwen2.5-32B-Instruct judge:

```bash
python -m src.inference.relabel_answer_judge --input <answers.jsonl> \
    --judge_model Qwen/Qwen2.5-32B-Instruct --judge_tag qwen32b
```

## 📁 Repository layout

```
src/
├── data/          # on-policy sampling, answer correction and SFT data
├── training/      # OP-SFT, KΔ-FT and FGPO (LLaMA-Factory / EasyR1 launchers)
├── inference/     # FoK scoring J = P(yes) / (P(yes) + P(no)) and vLLM inference
├── baselines/     # Verbal, Prob, N-Prob, Consistency, SemEnt, EliCal
└── configs/       # DeepSpeed configs
scripts/           # stage1_op_sft.sh, stage2_kdelta.sh, stage3_fgpo.sh and evaluation
easyr1_patches/    # FGPO reward, prompt and verl changes
llamafactory_patches/  # corrected-span SFT changes to LLaMA-Factory
```

## 📖 Citation

```bibtex
@inproceedings{fu2026feeling,
  title     = {Feeling of Knowing in Large Language Models},
  author    = {Fu, Zichuan and Wu, Xian and Gao, Jingtong and Zhang, Wenlin and Li, Jiaxuan and Wang, Binhao and Deng, Yimin and Li, Guojing and Li, Xiaopeng and Xu, Derong and Zheng, Yefeng and Zhao, Xiangyu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This code builds on [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), [EasyR1](https://github.com/hiyouga/EasyR1), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [vLLM](https://github.com/vllm-project/vllm) and [EliCal](https://github.com/Trustworthy-Information-Access/Annotation-Efficient-Universal-Honesty-Alignment).
