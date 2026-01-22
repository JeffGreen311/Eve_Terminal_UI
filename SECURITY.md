# Security Policy / Hardening Notes

## Do not commit secrets
This project intentionally contains **no** real credentials.
Use environment variables or a local `.env` file.

- `REPLICATE_API_TOKEN`
- `ELEVENLABS_API_KEY`
- Any other service keys or tokens

Commit only `.env.example`.

## Recommended secret scanning
Before publishing:
- Run a secret scanner (e.g., `gitleaks`, `trufflehog`) on the repo history.
- Rotate any keys that were previously committed.

## Auto-install is disabled by default
This project can optionally check dependencies, but it **will not** run `pip install` unless you opt in:

- `EVE_AUTO_INSTALL_DEPS=1`

## Safer defaults
- Output directories default to `~/.eve/*` to avoid leaking machine-specific paths.
- Network calls should fail fast if credentials are missing.

## Reporting vulnerabilities
If you discover a security issue, please open a private report to the maintainer.
