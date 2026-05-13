# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in `causal-gpt-rl`, please **do not
open a public issue or pull request**. Instead, report it privately by email:

- **junho@ccnets.org**

Include:

- A description of the issue and the affected version(s).
- Steps to reproduce, or a minimal proof of concept.
- Any suggested mitigation if you have one.

We will acknowledge receipt within 7 days and aim to provide a fix or
remediation plan within 30 days for confirmed issues. After a fix is
released, we will publicly disclose the issue along with credit to the
reporter (unless you request anonymity).

## Scope

This policy covers the `causal-gpt-rl` Python package (model loading,
inference runtime, bundle format).

Out of scope:

- Vulnerabilities in third-party dependencies (`torch`, `transformers`,
  `gymnasium`, etc.) — please report those to the respective projects.
- Misuse or accidental exposure of model bundles hosted on third-party
  platforms (e.g. Hugging Face Hub) — those are governed by the host's
  own policy.
