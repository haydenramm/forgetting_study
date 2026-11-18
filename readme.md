# Forgetting Study
Repository for an exploratory investigation of how and why transformer-based architectures forget and experience performance degradation on tasks. The experiments involve training a small transformer architecture on a series of toy tasks provided by the [MADLAB repository](https://github.com/athms/mad-lab).

We implement a toy transformer of our own making

## Experimental aims
- Observe where and when an attention head enforces / introduces sparsity during training
- Observe whether training on subsequent tasks produces orthogonal updates in parameter space; if this is the case, we can produce an empirical demonstration / explanation for transformer-based architectures' capacity for continual learning.
