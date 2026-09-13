# Dataset storage

Local datasets are grouped by application domain and excluded from Git. `textile/` contains fabric datasets; `general/` contains reusable industrial and auxiliary corpora such as MVTec and VisA.

Each imported dataset should document its source, license, download date, size, categories, annotation format, adapter, and split policy. Weights belong to a domain artifact cache, never inside a dataset directory.
