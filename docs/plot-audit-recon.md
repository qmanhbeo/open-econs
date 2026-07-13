# Plot-audit findings (2026-07-11)

> Moved from `ROADMAP.md` during the 2026-07-13 trim.

- `OLSResult.plot()` deprecated in v0.8, removal in v0.9 (panel 4 had been a
  `"planned for v0.4"` placeholder since v0.3; no OE diagnostics annotated;
  self-contained but redundant/generic).
- `EventStudyResult.plot()` kept as-is (not redundant with any dependency;
  self-contained; minimal but defensible domain convention).
- Replacement diagnostics-annotated plot logged as non-blocking roadmap item —
  the audit surfaced that `self.diagnostics()` + `self.condition_number` are
  already computed but `.plot()` never used them. The analysis/ingredients
  exist; only the annotation is missing.

> **Non-blocking** — Diagnostics-annotated `OLSResult.plot()` replacement:
> annotate the existing 4-panel layout with the test statistics/p-values
> `self.diagnostics()` already computes (JB p-value on the QQ panel, BP p-value
> on residuals-vs-fitted, DW stat as an annotation, condition number in the
> margin). Would flip this from "redundant wrapper" to "the one thing the audit
> found genuinely missing." Not blocking v0.8; can land in the same window or
> later.
