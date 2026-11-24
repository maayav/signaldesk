# Attribution and sources

## Dataset

SignalDesk uses **BANKING77**, released by PolyAI and introduced by Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić in 2020.

- [Dataset source at the pinned commit](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)
- [Upstream license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE): Creative Commons Attribution 4.0 International.
- [License description](https://creativecommons.org/licenses/by/4.0/)
- [Paper: Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/)

The original downloaded CSVs are unchanged. Training derives normalized exact-text keys, removes specified duplicate/overlap rows from the official training set, and creates four internal partitions. Trained artifacts contain learned parameters and selected labeled source examples. These transformations and examples remain subject to the applicable dataset attribution requirements. No affiliation with or endorsement by PolyAI or the paper’s authors is implied.

```bibtex
@inproceedings{casanueva-etal-2020-efficient,
  title = {Efficient Intent Detection with Dual Sentence Encoders},
  author = {Casanueva, I\~nigo and Tem\v{c}inas, Tadas and Gerz, Daniela
            and Henderson, Matthew and Vuli\'c, Ivan},
  booktitle = {Proceedings of the 2nd Workshop on Natural Language Processing
               for Conversational AI},
  year = {2020},
  pages = {38--45},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2020.nlp4convai-1.5},
  url = {https://aclanthology.org/2020.nlp4convai-1.5/}
}
```

## Project inspiration

[Build Your Own X](https://github.com/codecrafters-io/build-your-own-x) and [Project Based Learning](https://github.com/practical-tutorials/project-based-learning) inspired the learning-by-building approach. SignalDesk is an independent implementation; those collections are not sources of copied project code, and their maintainers are not affiliated with this repository.

## Software

The project uses Python, NumPy, SciPy, scikit-learn, pandas, joblib, FastAPI, Pydantic, Uvicorn, Streamlit, requests, Plotly, pytest, HTTPX, and Ruff. Their respective licenses continue to apply. Package versions are specified in `pyproject.toml` and the environment lock file.

Original project code and documentation are MIT licensed. The MIT license does not replace the dataset or dependency licenses.
