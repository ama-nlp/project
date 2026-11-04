# Big Brother is (Maybe) Watching

Advait Dintakurti, Aman Jayesh, Revan Pedaballi, Shreyas Ram (IIIT Hyderabad, ANLP).

Chain-of-thought monitoring assumes that a model's reasoning is connected to what it does.
We test that assumption in reward-hacking model organisms: UK AISI's OLMo-3.1-32B models,
LoRA-trained with RL on a CodeContests environment whose tests can be cheated. The study
began by asking whether a model hides its hack when told its reasoning is watched; it became
a study of whether the reasoning still controls the code after reward-hacking RL, and what in
training removes that control.

All experiments are inference only. We train no models; we change the prompt, the reasoning,
or the model's activations, and score the code with AISI's own grader.

## Findings so far

- **Telling the model it is watched changes nothing that matters.** No oversight framing
  (eight arms plus paraphrases, including penalties for a flagged hack and direct
  instructions to conceal) changed how often the model hacked or made its reasoning name the
  hack less often.
- **On the KL 0.02 organism the hack is chosen at the second output token.** The model's own
  probability of opening with the hack instead of `<thinking>` is calibrated and set by the
  prompt; the prompt's "don't hack" warning, not its description of the hacks, holds it back.
  Steering directions change *when* the model hacks, not *whether*, and a Jacobian lens finds
  no hack held silently in the reasoning.
- **A plan written into the reasoning steers the pre-RL model's code (94% of plans), and
  reward-hacking RL removes that control.** Without a KL penalty, no plan changes the code.
  Control is lost when the model learns to hack: AISI's no-hack control keeps it, and the
  collapse coincides with the reward jump in training. At a matched training step, a KL
  penalty preserves more control in both released seeds; a single-seed comparison suggests
  that the penalty on the *code* tokens is what matters.
- **Reasoning that reads as faithful need not control the action.** AISI's run trained for
  faithful reasoning is barely steered by plans written into it.
- **Replications in progress** (see the run log): the OLMo-3 7B organisms reproduce the
  E-track result (E10); AISI's Qwen2.5-32B organisms (E12), fresh draws of the single-seed
  runs (E11) and natural instead of template plans (E13) are running or queued.

Every result, decision rule and deviation is in
[`docs/results/interp-live.md`](docs/results/interp-live.md). Rules were written into it before
each run; measures defined afterwards are marked as such.

## Repository

| Path | What it holds |
|---|---|
| [`project/`](project/) | The Python package, scripts and Slurm jobs. **How to run anything is in [`project/README.md`](project/README.md).** |
| [`docs/results/interp-live.md`](docs/results/interp-live.md) | The interpretability log: plans, pre-stated rules, run log, findings (I and E tracks). |
| [`docs/results/`](docs/results/) | One record per behavioural run (001–031) and [`aisi-rh-summary.md`](docs/results/aisi-rh-summary.md). |
| [`docs/`](docs/) | Phase plan and gates ([`phases.md`](docs/phases.md)), phase records, the [proposal](docs/proposal.pdf), [`handoff.md`](docs/handoff.md). |
| [`labeller/`](labeller/) | The blinded human-labelling page for reasoning traces ([its README](labeller/README.md)). |
| [`site/`](site/) | The results site, built from `docs/` by `site/build.py` and deployed to GitHub Pages. |
| [`AGENTS.md`](AGENTS.md) | Conventions for anyone (or any agent) changing the code: data provenance, experimental contracts, testing rules. |

Active work happens on feature branches before it reaches `main`; the E10–E13 code is on
`olmo7b-etrack`.

## Running it

Everything runs from `project/`, locally for code and tests and on IIIT's Ada cluster for
models. Start with [`project/README.md`](project/README.md).

```bash
cd project
bash scripts/fetch_upstream.sh   # upstream evaluator and dataset, not redistributed here
uv run --group dev pytest -q     # offline, CPU-only test suite
```

## Credits

The model organisms, environment and grader are UK AISI's
([code](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment),
[models](https://huggingface.co/collections/ai-safety-institute/reward-hacking-mo-checkpoints-and-rollouts-6a44dcbbe6d9be7df096f698)).
The LeetCode evaluator used in the early phases comes from
[`ariahw/rl-rewardhacking`](https://github.com/ariahw/rl-rewardhacking) and is fetched, not
vendored.

## License

[MIT](LICENSE).
