# Stage 5 commands (repository root)

Use the existing environment. All runtime/cache/log/artifact writes remain under independent_validation_outputs/.

```sh
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python independent_validation_outputs/source/cli.py --audit
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python independent_validation_outputs/source/cli.py --prepare-external
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python independent_validation_outputs/source/cli.py --fit-frozen-research-model
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python independent_validation_outputs/source/cli.py --evaluate-external
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python independent_validation_outputs/source/cli.py --tests
```

The missing-PATH commands return NO_EXTERNAL_DATA. The explicit fit command is exercised but cannot train while no independent set and signed scientific-provenance review exist; it returns NOT_TRAINED. `--audit` never trains or scores external targets. `--tests` uses only offline tiny in-memory synthetic fixtures and mocks; no synthetic external performance files are produced.

For a future real, independently documented dataset staged inside the repository, replace the omitted PATH with `independent_validation_outputs/incoming/<actual-source>.csv` (no such observations are provided by this task) and supply its `.manifest.json` with inspectable evidence and signatures. Resolve development provenance with the row-specific review described in external_data_dictionary.md; do not fabricate the review or change labels. Prepare must pass the locked schema/overlap/provenance gates before explicitly requesting the at-most-25-fit research model. Evaluation requires that frozen model and consumes outcomes once. All commands fail closed on identity changes and pre-existing outcome access. Never load an untrusted externally supplied pickle/model.

```sh
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python independent_validation_outputs/source/run_stage5.py --prepare
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python independent_validation_outputs/source/run_stage5.py
```

These execute the fresh-kernel readiness notebook and all CLI checks. The first defers STAGE5_REPORT.md; the second writes it last and stops. Neither automatically acquires observations or trains a model. Positive gated fitting/evaluation paths are implemented but not empirically executed without eligible data; tests cover the guards, arithmetic and isolation rather than claiming a real external validation.
