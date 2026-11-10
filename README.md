# Synthetic credit-offer experiment

> **All data, effects, and results in this project are synthetic. They are teaching assumptions—not observations from customers, evidence from a live portfolio, or a recommendation for a real credit policy.**

This project demonstrates a reproducible 1:1 account-level randomized experiment for a simulated credit-line-increase offer. It compares 90-day spend with a serious-delinquency guardrail, including a sample-size approximation, covariate adjustment, subgroup uncertainty, A/A calibration, and the risk of repeated significance checks.

## Decision memo — one seeded simulation

With the configured 5,000 synthetic accounts and fixed seed, the simulated run produces:

- **Assignment check:** 2,497 treatment and 2,503 control accounts; the 50/50 sample-ratio-mismatch test has $p=0.932$.
- **Primary outcome:** treatment-minus-control 90-day spend is **+$20.50 per account** (95% CI $[+9.90,+31.10]$, $p=0.00015$). CUPED and OLS adjusted estimates are both about **+$17.55**.
- **Guardrail:** the 60+ DPD risk difference is **+0.08 percentage points**. Its one-sided 95% upper bound is **+1.78 pp**, above the illustrative **+0.5 pp** non-inferiority margin.
- **Decision:** **Do not ship based on this simulated experiment.** Spend is positive, but the run does not establish that the delinquency increase is below the predeclared margin. This is not proof that the true risk effect is harmful; the guardrail interval is too wide to meet the rule.

Risk-band estimates are exploratory. Only 179 accounts fall in the high-risk band (82 treatment, 97 control), leaving its spend interval very wide. Its observed delinquency contrast should not be mistaken for the planted high-risk effect or treated as confirmatory evidence.

## Power and monitoring demonstrations

Using the simulated control arm, the baseline spend mean is about $184.08 and variance about 29,165. The normal-approximation calculation for a 5% relative MDE, two-sided $\alpha=0.05$, and 80% target power gives about **5,405 accounts per arm** without adjustment. The observed CUPED variance ratio in this particular synthetic draw is 0.024, implying about **133 per arm** under that measured ratio. This unusually strong reduction is specific to the simulator and this draw; it is not a guaranteed planning credit. The primary sample-size calculation does not establish adequate guardrail power.

Across 300 seeded A/A repetitions, the observed false-positive rate was 4%. In the repeated-peeking demonstration, stopping at the first daily $p<0.05$ produced a 16.5% false-positive rate, versus 4% for the single-final-look comparator. These are Monte Carlo teaching results, not universal rates.

## Repository contents

- [config.py](config.py): shared simulation assumptions and seeds.
- [src/simulate.py](src/simulate.py): synthetic account generation, assignment, risk bands, and planted effects.
- [src/power.py](src/power.py): sample-size approximation and MDE curve.
- [src/analysis.py](src/analysis.py): SRM, unadjusted/CUPED/OLS estimates, guardrail analysis, risk-band effects, and decision rule.
- [src/pitfalls.py](src/pitfalls.py): A/A calibration and repeated-peeking demonstration.
- [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb): executable end-to-end narrative.
- [docs/design.md](docs/design.md): experiment design and staged implementation plan.

## Run and validate

The project uses Python with NumPy, pandas, SciPy, statsmodels, matplotlib, and pytest. Open and run [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb) from top to bottom for the analysis and plots. The notebook's final section runs the tests; the verified result for this revision was **12 passed**.

## Scope and limitations

- Treatment means offer assignment (intent-to-treat); acceptance and usage are not modeled.
- The experiment assumes no interference between accounts and observes outcomes over only 90 days.
- Risk scores and bands, spend/delinquency outcomes, and planted effects are synthetic and not empirically calibrated.
- Subgroup comparisons are exploratory, and no multiplicity correction is applied.
- The normal-approximation power result and observed CUPED variance reduction do not establish guardrail safety.
- A simulation-only “do not ship” result is not a policy recommendation for a real portfolio.
