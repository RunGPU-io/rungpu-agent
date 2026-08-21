# Changelog

User-visible changes to the RunGPU Agent. Internal platform implementation and
security details are intentionally excluded.

## [1.3.14] - 2026-08-20

- Added support for managed text-to-speech workloads, including optional
  reference-voice cloning.

## [1.3.12] - 2026-08-19

- Improved memory availability for managed generation workloads.

## [1.3.11] - 2026-08-19

- Prevented long managed generation steps from failing on transient status
  request timeouts.

## [1.3.10] - 2026-08-19

- Improved startup reliability for managed generation workloads.

## [1.3.9] - 2026-08-19

- Improved compatibility and validation for managed generation workloads.

## [1.3.8] - 2026-08-19

- Improved agent startup and managed workload reliability.
- Added release version reporting to simplify support and troubleshooting.

## [1.3.7] - 2026-08-19

- Fix mixed indentation in the embedded ComfyUI runner introduced in v1.3.6.
- Compile-check the exact normalized runner during tests to prevent recurrence.

## [1.3.6] - 2026-08-19

- Wait for the complete ComfyUI API to become ready before submitting managed
	workflows.
- Report HTTP, empty-response, invalid-response, and container-exit failures
	separately instead of showing a generic JSON parsing error.

## [1.3.5] - 2026-08-19

- Fixed model preparation on Windows when the verified model cache does not
	permit timestamp updates.

## [1.3.4] - 2026-08-19

- Tightened workload assignment validation and removed local runtime guessing.
- Made required managed-result upload failures fail the job instead of
	reporting completion without a retrievable result.

## [1.3.2] - 2026-08-18

- Added support for new image and video generation workloads.

## [1.3.1] - 2026-08-17

- Expanded support for managed AI generation workloads.

## [1.3.0] - 2026-08-16

- Improved workload security, reliability, and compatibility.
- Improved setup and asset handling.

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
