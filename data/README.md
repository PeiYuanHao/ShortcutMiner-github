# Local datasets

Raw research datasets do not belong in this Git repository. Store Waterbirds,
Camelyon17, Colorized-MNIST, and other real datasets locally, then point a dataset
configuration at the mounted root.

The root Docker Compose configuration mounts this directory at `/data` as
read-only. One possible local layout is:

```text
data/
├── waterbirds/
├── camelyon17/
└── colorized_mnist/
```

Only this README is tracked below `data/`; dataset contents are ignored. Tests
create their own small temporary fixtures and do not require these datasets.
