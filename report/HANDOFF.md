# Report Handoff — COMP0225 RAI Group Project Report 2

This file summarises the work done so a future Claude session (or a
human collaborator) can pick up the report writing without re-doing
the analysis. Read this first.

## Project context

- **Module**: COMP0225 — UCL Robotics and AI Group Project
- **Submission deadline**: 22 May 2026, 16:00 (via Moodle)
- **Team**: Vaibhav Mehra, Muhammad Maaz, Benjamin Li
- **Template**: IEEE conference (`IEEEtran`)
- **Length**: 10–12 pages including bibliography
- **Audience**: UCL CS graduate level
- **Bibliography**: 10–30 references (currently 12)

### Team responsibility split

- **Vaibhav** — Methodology §III-A (Synthetic Data Generation) and
  §III-B (Residual Crowd Motion Model). His TikZ figure for the
  residual model is already drafted.
- **Muhammad** *(this session)* — Methodology §III-A/-B that follows
  Vaibhav's (renumbered to §III-C and §III-D in the final report),
  Experimental Setup, Results.
- **Benjamin** — Safety layer: conformal-prediction shield plus a
  PPO-trained diffusion policy. Distinct from Muhammad's offline
  diffusion predictor — see the disambiguation note in §V.
- **Joint sections (still TODO)**: Introduction, Project Management,
  Conclusions.

## Current state of [main.tex](main.tex)

3 pages, clean compile, 12 citations. Contains Muhammad's complete
contribution:

- Abstract (~14 lines)
- Methodology §III opener + §III-C Trajectory Prediction Models (with
  4 subsubsections: Social-LSTM/+V, Social-GRU, Trajectory Transformer,
  Diffusion) + §III-D Transfer Learning via SDD Pretraining
- Experimental Setup §IV (one paragraph, dense)
- Results and Discussion §V (with 3 tables, 3 figures, all numerical
  claims grounded in `results_ft.json`)
- Empty section headers for the joint TODOs (Introduction, Project
  Management, Conclusions)

The build chain is `pdflatex → bibtex → pdflatex → pdflatex`. On
Overleaf the project must include `references.bib` in the root
directory and the `figures/` subfolder.

## Final numerical claims (verified against codebase)

### ETH/UCY (avg over 5 scenes)

| Model | ADE | FDE | minADE@20 | minFDE@20 | NLL |
|---|---|---|---|---|---|
| CV | 0.736 | 1.272 | 0.610 | 0.814 | — |
| Social-LSTM | 0.596 | 1.248 | 0.581 | 0.597 | 18.0 |
| SLSTM+V (ours) | 0.585 | 1.240 | 0.587 | 0.585 | 36.2 |
| Social-GRU (ours) | 0.586 | 1.238 | 0.596 | 0.924 | — |
| Transformer (ours) | **0.568** | 1.203 | 0.638 | **0.532** | **1.54** |
| Diffusion (ours) | 0.579 | **1.195** | 1.174 | 2.032 | 5.14 |
| Trajectron++ | 0.853 | 1.817 | **0.385** | 0.674 | 8.49 |

### SDD (avg over 8 scenes)

SLSTM+V wins on ADE (0.656), minADE@20 (0.755), minFDE@20 (0.604).
Transformer wins on FDE (1.326). Diffusion minADE@20 stays high (1.710).

### Fine-tune effect (scratch → SDD then ETH/UCY+synth FT)

Social-LSTM −9.9% (0.596 → 0.537), Social-GRU −6.3%, SLSTM+V −5.1%,
Transformer +0.4% (no benefit), Diffusion −7.6%.

Per-scene fine-tune effect (Social-LSTM): eth −5.9%, hotel −29.8%.

### Diffusion ablation

Removing the encoder detach on the Gaussian head cut ADE from
**1.37 m → 0.58 m** on ETH/UCY without losing sample diversity.

## Architecture hyperparameters (verified)

| Model | Key params |
|---|---|
| Social-LSTM | hidden 128, embedding 64, pool radius 2.0 m |
| SLSTM+V | same + velocity (2D→4D focal input) |
| Social-GRU | bidirectional GRU, 2×128→128, same pool |
| Transformer | Pre-LN, d_model 128, 4 heads, 2 layers/side, 48-token memory, 12 learned queries |
| Diffusion | shared Transformer encoder + Gaussian MLP head + denoiser, T=100, β linear [1e-4, 0.02], λ_DDPM=0.3, DDIM-50, K=20 |

### Training (universal)

- Adam, lr 1e-3, weight decay 1e-4, batch 64
- `ReduceLROnPlateau` patience 5, factor 0.5
- Scratch ETH/UCY ≤ 200 epochs (patience 20)
- SDD pretrain ≤ 100 epochs
- Fine-tune lr 1e-4, ≤ 50 epochs (patience 15, grad-clip 1.0)

### Datasets

- ETH/UCY: 5 scenes (eth, hotel, univ, zara1, zara2) at 2.5 Hz, 8 obs +
  12 pred steps (3.2 / 4.8 s)
- SDD: 8 scenes downsampled stride 12 from 30 fps to 2.5 Hz, pixels to
  metres at 0.0417 m/px, ~249k pedestrian sequences

## Figures currently in the report

All TikZ source in [figures/](figures/), all PNGs in figures/ too.

| Figure | Source | Notes |
|---|---|---|
| Fig. 1 — Transformer architecture | `figures/transformer.tex` | TikZ, scaled to 0.6 columnwidth |
| Fig. 2 — Two-stage transfer pipeline | `figures/finetune_pipeline.tex` | TikZ, scaled to 0.6 columnwidth |
| Fig. 3 — Qualitative trajectories on zara1 | `figures/fig03_qualitative.png` | matplotlib output |

Removed figures (still in figures/ but not referenced): `social_lstm.tex`,
`diffusion.tex`, `fig_nll.png`, `fig_generalisation.png`,
`fig02_scatter.png`, `fig04_diversity.png`.

If a future writer wants more figures, the cleanest matplotlib options
in [/plots/report/](../plots/report/) are `fig02_scatter.png`
(accuracy–diversity scatter, single panel, clean) and `fig_nll.png`
(NLL bar chart). The NLL one was dropped because the 12× advantage is
already discussed in prose and Table I.

## Decisions and style rules established this session

1. **No em dashes in prose** (`---` or `—`). Only allowed in table
   cells where `& --- \\` means "N/A".

2. **No bold in prose.** Bold is reserved for best-value cells in
   tables. Model name introductions are plain text.

3. **No italics in prose.** Scene names (eth, hotel, univ, zara1,
   zara2) and section labels are plain text. "(ours)" in tables is
   plain text. The only italics left are LaTeX-internal (`\textit{...}`
   is allowed in figure captions if needed).

4. **No `+` in prose body** except in proper names: `Trajectron++`,
   `SLSTM+V` are kept; `ETH/UCY + synthetic` becomes "ETH/UCY and
   synthetic" or "ETH/UCY plus synthetic".

5. **Model naming**: "Social-GRU" (not "GRU-v2") — the codebase class
   is `SocialGRUv2` and result-file keys are `gru_v2`, but the paper
   uses the cleaner name. Architectural description (bidirectional GRU)
   is correct either way.

6. **No `\to` arrows in prose**. Use "from X to Y" or "X falls to Y".
   Arrows are fine inside equations.

7. **Anti-LLM rules** (applied repeatedly across the session):
   - No "we observe / it can be seen / this demonstrates"
   - No "Furthermore / Moreover" as openers
   - No formulaic "X is the only model that genuinely..."
     constructions
   - Vary sentence length: mix short with long
   - Don't end every paragraph on a synthesis sentence
   - Avoid abstract subjects like "Attention sees..." — use concrete
     subjects ("the variance head reads...")
   - Replace personification ("X dominates", "X wins on Y, loses on Z")
     with concrete numerical claims
   - Drop generic adjectives ("significantly", "considerably") in
     favour of the actual % or numbers
   - Words to avoid as filler: genuinely, particularly, notably,
     remarkably, essentially, fundamentally

## What was deliberately not done

- **No SOTA comparison paragraph in the report.** The literature
  context (LED 0.21 minADE@20, MID 0.27, etc.) was discussed in chat
  but not written in. A short "Relation to prior work" paragraph at
  the start of §V would strengthen the 15% Background-rubric criterion
  and is the single highest-leverage addition still available.

- **No new analysis or numbers.** Every number in the report is
  grounded in `results_ft.json` and the training scripts. Don't invent
  new claims.

- **No structural restructuring.** The section order follows the
  module's suggested structure exactly.

## Open items for the joint sections

The three TODO sections are placeholders in main.tex:

- **§I Introduction** — background, problem definition, contributions.
  Should reference Report 1 lightly but not re-do the lit review.
- **§II Project Management** — roughly 1 page covering milestones,
  timeline, roles, risks.
- **§VII Conclusions** — roughly 1 page covering summary, challenges,
  insights, limitations, lessons.

The Methodology section needs Vaibhav's two subsections (Synthetic
Data Generation, Residual Crowd Motion Model) inserted **before**
Muhammad's `\subsection{Trajectory Prediction Models}` at line ~70.

There are two unresolved `\ref{sec:...}`-style references in
Muhammad's prose that point to Vaibhav's labels (`sec:rcmm`,
`sec:synth`) — these were rewritten as plain text earlier in the
session and should now be safe, but double-check after integration.

## Code base

- Repository root: `/cs/student/projects1/2023/muhamaaz/year-long/`
- Environment: `source crowdnav-env/bin/activate` (Python 3.9.25,
  PyTorch 2.5.1+cu121, RTX 3090 Ti)
- Models in [models/](../models/)
- Training scripts: `models/train_*.py`, `models/finetune_eth.py`
- Evaluation: `evaluate_all.py` (5-model comparison on ETH/UCY)
- Results JSONs: `results.json`, `results_ft.json` at repo root

## Compile notes

- The build chain is `pdflatex → bibtex → pdflatex → pdflatex`.
- On Overleaf: ensure `references.bib` is uploaded to the project root
  and the `figures/` folder is uploaded alongside `main.tex`. If
  citations show as `[?]`, BibTeX hasn't run — recompile from scratch.
- `figures/transformer.tex` and `figures/finetune_pipeline.tex` use
  `\resizebox{0.6\columnwidth}{!}{\input{...}}` to keep diagrams
  compact.

## Quick sanity-check before submission

1. `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
2. `pdfinfo main.pdf | grep Pages` — must be 10–12 once joint sections
   are filled in.
3. `grep -nE " --- | — " main.tex | grep -v "& ---"` — must be empty.
4. `grep -c "\\\\cite" main.tex` — should be 10–30.
5. All `\ref{}` calls resolve (no `??` in the rendered PDF).
6. Read aloud test on Muhammad's sections — no LLM rhythm.
