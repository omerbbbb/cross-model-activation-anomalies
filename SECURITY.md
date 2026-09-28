# Security

## Public-release posture

This repository does not require API keys, Hugging Face tokens, cloud credentials,
or private datasets for the documented experiments.

Model loading in the maintained Python code uses:

```python
trust_remote_code=False
```

The public research notebooks have also been sanitized so historical cells do not
enable remote model-repository code execution.

## Third-party downloads

The experiments download public model weights/tokenizers and the AG News dataset
from third-party repositories. Run the project in an isolated virtual environment
and review dependency/model sources according to your own threat model.

## Secrets

Do not commit `.env` files, private keys, credentials JSON files, or access tokens.
The repository `.gitignore` includes common secret-file patterns as an additional
guard, but it is not a substitute for checking staged changes before pushing.

A useful pre-push check is:

```bash
git diff --cached
git grep -nE 'sk-|hf_|github_pat_|ghp_|AKIA'
```

## Historical results

Some notebook outputs document failed or superseded experiments. In particular,
the original BLOOM float16 result was later invalidated by a NaN diagnostic and is
retained only as research history. Corrected BLOOM results use float32 and are
identified in the README and results files.


## Numerical validity guard

The maintained hidden-state extraction code checks for non-finite activations.
If a model produces NaN or Inf hidden states, scoring stops with a
`FloatingPointError` rather than producing a percentile from invalid values.
