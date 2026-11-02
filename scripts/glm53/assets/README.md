# GLM-5.3 Flash serving assets

`mtp-draft-vocab-98304.i32` is a draft-head shortlist containing 98,304 unique
little-endian signed 32-bit global token IDs. It is 393,216 bytes, SHA256
`8e83de6161b0f75c212f7079b00e10af2c03701ae17afe62ee48bca35717e3e6`.
The target still verifies and samples from its full vocabulary. Byte tokens
0–255 and special tokens 154820–154855 are retained; padded IDs are excluded.
This asset is specific to the tokenizer and model identified in the
[serving recipe](../../../docs/backend/GLM53-SERVING.md).

The shortlist was calibrated exclusively on public text: the 20231101 English
and Chinese snapshots of
[Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
Python's `v3.13.0` standard library, TypeScript's `v5.7.3` source, and generated
tool-call syntax using seed 530. Every fifth document was held out. Ranking sums
each category's relative token frequencies, breaks ties by token ID, and retains
mandatory IDs first. The binary contains token IDs, no corpus text or user
conversations. The calibration corpus and generator are not included here.

Recorded tokenizer revision: `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`.
Tokenizer JSON SHA256:
`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`.
The tokenizer was checked against the served GGUF. Calibration corpus SHA256:
`cd67c5a5387a1120fd02bba160d9b1ff7b0151a80234402b48f6ed2dc77462de`.

Recorded held-out token coverage is 99.417% English, 99.678% Chinese, 99.347%
code, and 100% generated tool syntax. These measure token coverage, not draft
acceptance, coding quality, or serving speed.

The paired [chat template](../../../models/templates/GLM-5.3-Flash.jinja)
is 12,782 bytes, SHA256
`bb443c2772691a62b661da6515637c3ff70b2a790bf939b30220e337a6d9e723`.
It preserves supplied reasoning by default and renders deferred tool discovery
where the corresponding tool result occurs. Both published assets are exact
copies of the serving inputs; the launcher verifies their hashes before use.
