# Tasks

## Repository Hygiene

- [x] Keep source code, configs, and docs in git
- [x] Ignore local results, models, TensorBoard logs, simulator binaries, and local ffmpeg bundles
- [ ] Keep documentation aligned with current CLI and artifact paths
- [ ] Push only replayable source state, not bulky experiment outputs

## Rule-Based Pipeline

- [x] Keep `torcs_jm_par.py` as the protocol reference
- [x] Maintain a working `driver.py`
- [x] Keep JSON-backed rule-based tuning in `research_driver.py`
- [x] Maintain `fastest.py` for replaying the canonical best setup
- [x] Maintain `autoresearch.py` for full-lap search
- [x] Maintain `autoresearch.py --strategy documented-turns` for sector-aware search
- [x] Save per-run configs, summaries, replay commands, and best telemetry under `results/autoresearch/`
- [x] Preserve canonical best files in `autoresearch_best.json` and `autoresearch_best_result.json`
- [x] Replay saved configs with `record_best_run.py --config-path ...`
- [ ] Continue longer documented-turns runs for incremental lap-time gains
- [ ] Compare sector bottlenecks across the latest saved runs
- [ ] Freeze a submission-ready rule-based best lap and record it cleanly

## Karpathy-Style Autoresearch Follow-Up

- [ ] Try a Karpathy-style autoresearch workflow after the current `autoresearch.py --strategy documented-turns` path stabilizes.
- [ ] Do not reset tuning progress for that experiment. Use the current canonical `autoresearch_best.json` and `autoresearch_best_result.json` as the baseline.
- [ ] Add a markdown-driven experiment protocol similar to `program.md`, where the human defines the research rules and the agent follows a fixed loop.
- [ ] Keep the TORCS evaluation harness fixed, analogous to Karpathy's immutable `prepare.py`.
- [ ] Restrict the mutable surface to the experiment strategy or candidate generator, analogous to Karpathy's mutable `train.py`.
- [ ] Start the new `results.tsv` style log with the current best lap as the first `keep` baseline row.
- [ ] Compare the Karpathy-style workflow against the existing deterministic tuner only after it can replay and improve from the same best-known setup.

## RL Pipeline

- [x] Keep `torcs_env.py` compatible with Gymnasium
- [x] Keep reward shaping isolated in `reward.py`
- [x] Maintain checkpointed SAC training with resume support
- [x] Maintain `eval_sac.py` for verification runs
- [x] Maintain `record_best_rl_run.py` for telemetry and video capture
- [ ] Run longer SAC training blocks after rule-based tuning stabilizes
- [ ] Compare RL best laps against the canonical rule-based best
- [ ] Record and archive the best verified RL lap

## Submission Readiness

- [ ] Keep a reproducible command for replaying the best rule-based lap
- [ ] Keep a reproducible command for replaying the best RL lap
- [ ] Capture final telemetry for the chosen submission candidate
- [ ] Capture final video for the chosen submission candidate
- [ ] Sanity-check the public repo from a fresh clone without local artifacts
