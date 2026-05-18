# Report 2 Context

This file is the working handoff for the complete group draft. It does
not replace `HANDOFF.md`; it adds the professor instructions, team
framing, and missing asset checklist needed to finish Report 2.

## Professor / Module Instructions

- Module: COMP0225 RAI Group Project.
- Deadline: 22 May 2026, 16:00 via Moodle.
- Required format: LaTeX using the IEEE Transactions / IEEEtran template.
- Required length: 10-12 pages including figures, tables, and bibliography.
- Bibliography: 10-30 references.
- Audience: UCL CS graduate level.
- Report 2 focus: design, implementation, and evaluation of the proposed
  approach so far.
- The earlier literature review / initial survey can be referenced briefly to
  motivate choices, but the literature review should not be repeated in detail.
- The report is a joint effort and must clearly integrate the contributions of
  all group members.

## Grading Rubric

- Background and Problem Understanding: 15%.
  Clear problem and objectives, relevant background and initial-survey
  motivation, appropriate literature, independent problem formulation.
- Research Design, Objectives, Methodology, Reasoning: 30%.
  Clear objectives, justified methodology, quality of system design and
  implementation, logical technical reasoning.
- Research Outcomes, Results, Analysis, Conclusions: 40%.
  Appropriate experiments, clear and correct results, deep analysis, strengths,
  limitations, challenges, and valid conclusions.
- Communication, Structure, Presentation, Referencing: 15%.
  Clear organisation, academic style, effective figures/tables, proper
  references.

## Required Report Structure

1. Abstract: 150-200 words, self-contained.
2. Introduction: concise background, problem definition, approach, contributions.
3. Project Management: milestones, objectives, roles, risks, mitigation.
4. Methodology and System Design: architecture, algorithms, implementation,
   justified design choices.
5. Experimental Setup: datasets, environments, metrics, baselines.
6. Results and Discussion: figures/tables, analysis, comparison with
   expectations, strengths, limitations, challenges.
7. Conclusions: completed work, major challenges, insights, limitations,
   lessons learned.
8. Bibliography.

## Source Files

- `report/main.tex`: Muhammad's methodology/results draft. Do not overwrite.
- `report/HANDOFF.md`: verified result claims and writing style constraints.
- `PROGRESS.md`: project progress and final result summaries.
- `RESULTS.md`: detailed ETH/UCY result tables and interpretation.
- Prompt-pasted initial survey: use only as concise motivation.
- Prompt-pasted Vaibhav report: source for synthetic data generation, residual
  crowd world model, and their experimental setup/results.

## Team Framing

The complete report should describe a prediction-centered crowd navigation
system with three connected contributions:

- Vaibhav: Social-CVAE synthetic trajectory generation and residual crowd world
  model.
- Muhammad: trajectory predictor benchmark, SDD pretraining, ETH/UCY
  fine-tuning, and analysis.
- Benjamin: downstream safety/planning interface through conformal/PPO
  safety-planning work. Present his contribution as the completed downstream
  interface/design that consumes predictor outputs and proposes or verifies
  actions. Unless final numerical closed-loop results are provided, do not
  invent collision-rate or time-to-goal results.

## World Model / Vaibhav Contribution

Vaibhav's work is not just data generation. It has two parts:

- Social-CVAE synthetic trajectory generator.
- Residual crowd world model.

The world model predicts future pedestrian motion using constant velocity plus
a learned residual correction:

`Y_hat = Y_CV + R_theta(X, S)`

It should be presented as a planning-facing model because robot planning needs
forecasts of how nearby pedestrians may move.

Verified residual result on held-out `eth`:

- CV ADE/FDE: `1.0755 / 2.2819`.
- Simple Residual Model ADE/FDE: `0.9514 / 1.9171`.
- Improvement: `11.5% ADE`, `16.0% FDE`.
- Social Residual Model: `1.0069 / 2.0752`.
- Latent Crowd World Model: worse than CV.

Main interpretation:

- residual learning helped,
- extra social/latent complexity did not automatically improve generalisation,
- early stopping mattered,
- evaluation is limited to held-out `eth`.

## Verified Muhammad Results

ETH/UCY average over five scenes:

- CV: ADE `0.736`, FDE `1.272`, minADE@20 `0.610`, minFDE@20 `0.814`.
- Social-LSTM: ADE `0.596`, FDE `1.248`, minADE@20 `0.581`,
  minFDE@20 `0.597`, NLL `18.0`.
- SLSTM+V: ADE `0.585`, FDE `1.240`, minADE@20 `0.587`,
  minFDE@20 `0.585`, NLL `36.2`.
- Social-GRU: ADE `0.586`, FDE `1.238`, minADE@20 `0.596`,
  minFDE@20 `0.924`.
- Transformer: ADE `0.568`, FDE `1.203`, minADE@20 `0.638`,
  minFDE@20 `0.532`, NLL `1.54`.
- Diffusion: ADE `0.579`, FDE `1.195`, minADE@20 `1.174`,
  minFDE@20 `2.032`, NLL `5.14`.
- Trajectron++: ADE `0.853`, FDE `1.817`, minADE@20 `0.385`,
  minFDE@20 `0.674`, NLL `8.49`.

Fine-tuning effect, scratch to SDD then ETH/UCY plus synthetic:

- Social-LSTM: `0.596 -> 0.537`, `-9.9%`.
- SLSTM+V: `0.585 -> 0.555`, `-5.1%`.
- Social-GRU: `0.586 -> 0.549`, `-6.3%`.
- Transformer: `0.568 -> 0.570`, `+0.4%`.
- Diffusion: `0.579 -> 0.535`, `-7.6%`.

## Missing Assets From Teammate

These PNGs are referenced in Vaibhav's draft but are not currently present in
`report/figures/`. If the teammate supplies them, they can be included if the
page budget allows:

- `seed_diversity_grid_0_3.png`: generated trajectory diversity.
- `distribution_comparison.png`: real vs synthetic motion distributions.
- `filtering_comparison.png`: effect of angle filtering.
- `simple_world_model_eth_curves.png`: simple world-model training curve.
- `social_residual_world_model_eth_curves.png`: social residual training curve.

Priority if added:

1. `distribution_comparison.png`
2. `filtering_comparison.png`
3. residual training curves
4. diversity grid

If any are absent during compile, do not reference them from
`report2_complete.tex`.

## Page Budget

Target 10-12 pages:

- Abstract and introduction: about 1 page.
- Project management: about 1 page.
- Methodology and system design: about 3 pages.
- Experimental setup: about 1.5 pages.
- Results and discussion: about 3.5-4 pages.
- Conclusions: about 1 page.
- Bibliography: about 1 page.

If over 12 pages, remove optional PNG figures first. Do not remove the world
model section.

## Milestone Status Wording

For the submission draft, all milestone rows should say `Complete`, because the
report is written from the perspective of the submitted work package. Be precise
in prose: prediction/world-model evaluation is numerically complete; the
Benjamin safety/planning component is complete as an interface/design in the
current report; full closed-loop navigation metrics remain future validation
unless teammate data is supplied.

## Compile Checklist

From `report/`:

```bash
pdflatex -interaction=nonstopmode report2_complete.tex
bibtex report2_complete
pdflatex -interaction=nonstopmode report2_complete.tex
pdflatex -interaction=nonstopmode report2_complete.tex
pdfinfo report2_complete.pdf | grep Pages
```

Check:

- `report2_complete.pdf` exists.
- Page count is 10-12.
- No missing figures.
- No unresolved citations or references.
- Bibliography has 10-30 entries.
- `report/main.tex` remains unchanged.
