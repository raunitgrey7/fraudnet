"""fraudnet: graph machine learning for payment fraud.

The package is split by responsibility: ``sim`` generates a seeded payments world with planted fraud,
``features`` holds the point-in-time online feature store and the graph snapshots, ``models`` holds the
rules baseline, gradient boosting and the GraphSAGE network, ``evaluation`` holds metrics and the cost
model, and ``serving`` exposes the scoring API. ``bench`` ties them together into one reproducible run.
"""

import os

# Native thread pools (OpenMP in scikit-learn, intra-op threads in PyTorch) default to two threads unless the
# caller set OMP_NUM_THREADS. The benchmark and the API share the machine with other work, and single-row
# tree prediction gets slower, not faster, when every core is woken for it (measured in REPORT.md).
os.environ.setdefault("OMP_NUM_THREADS", "2")

__version__ = "0.1.0"
