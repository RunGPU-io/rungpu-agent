# Changelog

User-visible changes to the RunGPU Agent. Internal platform implementation and
security details are intentionally excluded.

## [Unreleased]

## [1.3.1] - 2026-08-17

- Added support for pool-managed ComfyUI model assets. Verified staged models
	are mounted read-only for the assigned job while existing model storage
	remains available for jobs without staged assets.

## [1.3.0] - 2026-08-16

- Hardened assignment validation and generated-file handling.
- Restricted managed downloads, redirects, and result uploads to approved
	HTTPS destinations.
- Isolated batch inference containers from outbound networks while preserving
	explicit workspace networking.
- Added SHA-256 verification and content-addressed caching for custom assets,
	and rejected model formats capable of unsafe deserialization.
- Removed runtime-triggered software installation. Runtime requirements are
	now installed only through the explicit `setup` command.
- Added compatibility checks for versioned coordinator job messages and
	provenance attestations for release artifacts.

## [1.2.1] - 2026-08-16

- Added optional public GPU contribution at no charge. Hosts can contribute
	without payout setup, or choose paid hosting after completing Stripe setup.
- Expanded `status` with enrollment, runtime readiness, and capability
	diagnostics plus an actionable setup recommendation.
- Added installation troubleshooting for Docker, WSL 2, security software,
	enrollment, permissions, and connectivity.
- Removed an incorrect Unix file-permission warning on Windows.

## [1.2.0] - 2026-08-16

- Added an explicit `setup` command for installing supported runtime
	requirements before starting the agent.
- Improved enrollment recovery by saving credentials before runtime checks and
	printing the exact next commands.
- Added runtime capability reporting so incomplete hosts are not marked ready.

## [1.1.0] - 2026-08-14

- Simplified setup for individual hosts and larger GPU fleets.
- Improved privacy by reducing unnecessary host information sent by the agent.
- Improved job reliability, recovery, and multi-GPU operation.
- Improved local storage cleanup and cache management.
- Improved service installation across Linux, macOS, and Windows.
- Expanded end-to-end runtime compatibility testing.
- General security, stability, and diagnostics improvements.

## [1.0.0] - 2026-07-28

Initial release of the RunGPU GPU Agent.
