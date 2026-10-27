# M3 references: verification log

Each reference cited in the M3 deck was opened on 2026-10-27 and its title, authors, venue, year and
identifier were checked against the publisher's page (or Crossref/BibTeX metadata from it).
Numbers quoted from a source are listed with their location.

| # | Reference (IEEE) | Checked at | Result |
|---|---|---|---|
| 1 | S. L. Garfinkel, "De-Identification of Personal Information," NISTIR 8053, NIST, Oct. 2015. | https://doi.org/10.6028/NIST.IR.8053 (Crossref record and PDF) | title and publisher match |
| 2 | Presidio documentation (presidio-analyzer 2.2.364 used). | https://presidio.dataprivacystack.org/ | page live |
| 3 | Google Cloud, "Sensitive Data Protection pricing." | https://cloud.google.com/sensitive-data-protection/pricing | live; content inspection US$3.00 per GiB from 1 GiB to 1 TiB per month, first 1 GiB free; content transformation US$2.00 per GiB; minimum of 1 KB billed per request for hybrid inspection. Because per-request minimums depend on batching, no total dollar figure is computed. |
| 4 | A. Riabi, M. Mahamdi, V. Mouilleron, D. Seddah, "Cloaked Classifiers: Pseudonymization Strategies on Sensitive Classification Tasks," Proc. 5th Workshop on Privacy in NLP, ACL, 2024, pp. 123-136. | https://aclanthology.org/2024.privatenlp-1.13/ (BibTeX) | match |
| 5 | PolyAI, BANKING77 dataset, CC BY 4.0. | https://huggingface.co/datasets/PolyAI/banking77 | page live |
| 6 | Faker documentation, MIT license (Faker 37.12.0 used). | https://faker.readthedocs.io/en/stable/ | page live (site shows a newer release; the project pins 37.12.0) |
| 7 | S553 lecture modules, Fall 2026. | course materials | lecture numbers as used in the deck |
| 8 | I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, I. Vulić, "Efficient Intent Detection with Dual Sentence Encoders," Proc. 2nd Workshop on NLP for Conversational AI, ACL, 2020, pp. 38-45. | https://aclanthology.org/2020.nlp4convai-1.5/ (BibTeX + PDF) | match. Table 3, BANKING77, full training data, accuracy: BERT-FIXED 87.19, BERT-TUNED 93.66, USE+ConveRT 93.36. |
| 9 | C. Morris, PrivacyGuard repository, 2026. | https://github.com/cmorris714/PrivacyGuard | live |
| 10 | J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," Proc. NAACL-HLT 2019, pp. 4171-4186. | https://aclanthology.org/N19-1423/ (BibTeX) | match |
| 11 | V. Sanh, L. Debut, J. Chaumond, T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019. | https://arxiv.org/abs/1910.01108 | match |
| 12 | W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, M. Zhou, "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers," Advances in NeurIPS 33, 2020. | https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html ; arXiv:2002.10957 | match |
| 13 | N. Reimers, I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," Proc. EMNLP-IJCNLP 2019, pp. 3982-3992. | https://aclanthology.org/D19-1410/ (BibTeX) | match |
| 14 | I. Pilán, P. Lison, L. Øvrelid, A. Papadopoulou, D. Sánchez, M. Batet, "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization," Computational Linguistics, vol. 48, no. 4, pp. 1053-1101, 2022. | https://aclanthology.org/2022.cl-4.19/ (BibTeX) | match |
| 15 | Explosion, spaCy English pipelines: en_core_web_lg and en_core_web_trf 3.8.0, MIT license. | https://spacy.io/models/en | page live; both pipelines listed |
| 16 | T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," Proc. EMNLP 2020: System Demonstrations, pp. 38-45. | https://aclanthology.org/2020.emnlp-demos.6/ (BibTeX) | match |
| 17 | F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, vol. 12, pp. 2825-2830, 2011. | https://jmlr.org/papers/v12/pedregosa11a.html | match |
| 18 | T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," Neural Computation, vol. 10, no. 7, pp. 1895-1923, 1998. | Crossref record for doi:10.1162/089976698300017197 | match |
| 19 | sentence-transformers/all-MiniLM-L6-v2 model card, Apache-2.0. | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | live; license apache-2.0 |
| 20 | dslim/bert-base-NER model card, MIT. | https://huggingface.co/dslim/bert-base-NER | live; license mit |
