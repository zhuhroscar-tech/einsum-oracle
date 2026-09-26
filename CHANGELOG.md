# Changelog

All notable source releases for `einsum-oracle` are tracked here.

## v0.1.5 - 2026-09-26

- Added package metadata links for the project homepage, issue tracker, and changelog so built distributions point users to maintenance resources.
- Made release-tag CI coverage explicit for `v*` tags and added repository-contract coverage for that release workflow wiring.

## v0.1.4 - 2026-09-24

- Added repository-contract coverage for required project files, README release links, CI artifact generation, CodeQL, and package/runtime version parity.
- Linked the release history from the README so users can verify what changed between published tags.

## v0.1.3 - 2026-09-24

- Modernized package license metadata to the current SPDX string format.
- Included the MIT license file in built distributions and raised the setuptools floor to a version that supports the metadata.
- Added regression coverage for package/runtime version parity and license metadata.

## v0.1.2 - 2026-09-24

- Corrected the release version after the v0.1.1 tag already existed.
- Rebuilt wheel and source distributions with the current package metadata.

## v0.1.1 - 2026-09-24

- Maintenance release with package metadata and test-suite checks carried forward from the initial implementation.

## v0.1.0 - 2026-09-17

- Initial public release of the exact dynamic-programming oracle for checking `numpy.einsum_path` contraction-order optimality.
- Added CLI fixtures reproducing known NumPy optimal-path regressions and baseline optimal cases.
