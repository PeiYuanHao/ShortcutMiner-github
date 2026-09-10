# Segmentation Providers

Segmentation is an optional image-construction service, not an Agent, shortcut detector, OOD generator, or scientific judge. A background intervention consists of a Candidate-derived policy, a valid foreground mask and an allowlisted replacement transformation.

`SegmentationProvider` accepts a bounded approved image and returns a mask plus coverage, provider/version, status and limitations. `ProvidedMaskSegmentationProvider` is the real CPU/offline implementation. It requires a same-size binary 0/255 mask under the approved dataset/run boundary. Background replacement preserves foreground pixels and modifies only the complement; constant, blurred-original and approved-sample replacements are supported.

`SAMCompatibleSegmentationProvider` is currently an unavailable local-only extension boundary. SAM, MobileSAM and SAM2 are not base dependencies. ShortcutMiner never installs them, downloads a checkpoint, uses Torch Hub, or performs remote segmentation. A future adapter must use a static allowlisted implementation, an explicit approved local checkpoint and bounded images. The optional real smoke runs only when `SHORTCUTMINER_REAL_SAM_TEST=1` and `SHORTCUTMINER_SAM_CHECKPOINT` names an existing local file; default skip is expected.
