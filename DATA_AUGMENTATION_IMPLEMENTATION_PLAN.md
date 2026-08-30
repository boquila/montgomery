# Native Ultralytics data augmentation

Status: implemented. The native deterministic augmentation subsystem is shipped; this file records
only the remaining release gates.

Implemented:

- task-specific detect/segment/classification pipelines with Ultralytics-pinned behavior;
- deterministic RNG, replay traces, letterbox/resize/perspective/HSV/flips;
- Mosaic, MixUp, CutMix, CopyPaste, classification policies, masks, formatting, and collation;
- training integration, compatibility documentation, provenance notices, and fixture tooling.

Remaining release gates:

- exhaustive cross-language numerical fixture matrix for every transform;
- threaded-loader throughput, GPU starvation, resume-next-batch, portability, and memory benchmarks;
- native AutoAugment/AugMix support (currently rejected explicitly as unsupported).

Generated fixtures and debug images belong under `target/`.
