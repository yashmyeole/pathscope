# PathVQA — Phase 0 dataset statistics

Source: `flaviagiammarino/path-vqa`

| Split | Q/A rows | Unique images | Q per image | Yes/No % | Median answer (words) |
|---|---|---|---|---|---|
| train | 19,654 | 2,599 | 7.56 | 49.61 | 1.0 |
| validation | 6,259 | 832 | 7.52 | 49.93 | 1.0 |
| test | 6,719 | 858 | 7.83 | 50.04 | 1.0 |

**Total:** 32,632 Q/A rows over 4,288 unique images; 49.76% yes/no.

## Train vs test overlap

- Question strings shared: 781 (26.85% of unique test questions; 67.24% of test rows)
- Exact (question, answer) pairs shared: 1,092 (29.86% of unique test pairs)
- Exact duplicate images (pixel MD5): 0 (0.0% of test images)

## Most common questions

- `what is present` × 6,820
- `what does this image show` × 3,015
- `where is this` × 1,151
- `is cardiovascular present` × 279
- `is oral present` × 270
- `where does this belong to` × 253
- `where is this from` × 247
- `is endocrine present` × 215
- `is gastrointestinal present` × 214
- `is hepatobiliary present` × 210

## Images with at least one open-ended answer

- test: 511 (59.56%)
- train: 1,558 (59.95%)
- validation: 494 (59.38%)
