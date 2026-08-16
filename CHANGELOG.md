# Changelog

User-visible changes to the RunGPU Agent. Internal platform implementation and
security details are intentionally excluded.

## [Unreleased]

- Added optional public GPU contribution at no charge. Hosts can contribute
	without payout setup, or choose paid hosting after completing Stripe setup.

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
