# BANKING77 data

SignalDesk uses [BANKING77 from PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data), introduced by Casanueva et al. in [Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/), 2020.

The public release contains 10,003 training queries and 3,080 test queries labeled with 77 online-banking support intents. The download command retrieves the original CSV bytes from the immutable commit `57ec275d8078af65b7731c2a98be812d844a6d6b`:

```bash
python -m signaldesk.data --data-dir data/raw
```

| File | SHA-256 |
| --- | --- |
| `train.csv` | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `test.csv` | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

Source files retain the upstream `text,category` schema. Training performs normalized exact-text deduplication and records excluded rows and partition counts in artifact metadata. Those transformations do not alter the original source CSVs. The evaluation protocol is in [the model card](../docs/model-card.md).

The dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The [upstream license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE) and [citation](../docs/ATTRIBUTION.md) apply to redistributed examples and derived artifacts. The project’s MIT code license does not relicense the dataset.
