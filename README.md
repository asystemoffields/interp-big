# interp-big → CircuitAtlas

Work in progress on the `mission` branch.

**CircuitAtlas** learns a sparse dictionary over per-token attribution-graph edge flows. The result is a model-wide catalogue of reusable, causally validated mechanisms, plus a census of how much of a model's computation those mechanisms explain.

- [`SPEC.md`](SPEC.md) is the build spec: method, experiments, pre-registration rules and milestones.
- [`docs/design/`](docs/design/) is the research record behind the choice:
  - `surveys.json`: 18 proposals from 6 research angles;
  - `selection.json`: the judge-panel ranking and the red-team verdicts.
