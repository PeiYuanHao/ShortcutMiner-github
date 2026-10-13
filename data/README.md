# Local datasets

Raw research datasets do not belong in this Git repository. Store Waterbirds,
Camelyon17, Colorized-MNIST, and other real datasets locally, then point a dataset
configuration at the local dataset root.

Formal Windows experiments use `E:/ShortcutMiner-Experiments`, treated as
read-only. One possible local layout is:

```text
E:/ShortcutMiner-Experiments/
├── waterbirds/
├── camelyon17/
└── colorized_mnist/
```

Only this README is tracked below `data/`; dataset contents are ignored. Tests
create their own small temporary fixtures and do not require these datasets.
