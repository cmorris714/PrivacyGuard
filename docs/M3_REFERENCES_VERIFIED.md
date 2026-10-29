# M3 references: verification log

Each reference cited in the M3 deck was opened on 2026-10-27 (references 6, 19 and 21-23 on 2026-10-29) and
its title, authors, venue, year and identifier were checked against the publisher's page (or Crossref/BibTeX
metadata from it).
Numbers quoted from a source are listed with their location.

| # | Reference (IEEE) | Checked at | Result |
|---|---|---|---|
| 1 | PolyAI, BANKING77 dataset, CC BY 4.0. | https://huggingface.co/datasets/PolyAI/banking77 | page live |
| 2 | Faker documentation, MIT license (Faker 37.12.0 used). | https://faker.readthedocs.io/en/stable/ | page live (site shows a newer release; the project pins 37.12.0) |
| 3 | C. Morris, PrivacyGuard repository, 2026. | https://github.com/cmorris714/PrivacyGuard | live |
| 4 | I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, I. Vulić, "Efficient Intent Detection with Dual Sentence Encoders," Proc. 2nd Workshop on NLP for Conversational AI, ACL, 2020, pp. 38-45. | https://aclanthology.org/2020.nlp4convai-1.5/ (BibTeX + PDF) | match. Table 3, BANKING77, full training data, accuracy: BERT-FIXED 87.19, BERT-TUNED 93.66, USE+ConveRT 93.36. |
| 5 | S. L. Garfinkel, "De-Identification of Personal Information," NISTIR 8053, NIST, Oct. 2015. | https://doi.org/10.6028/NIST.IR.8053 (Crossref record and PDF) | title and publisher match |
| 6 | IBM, "Cost of a Data Breach Report 2026," 2026. | https://www.ibm.com/reports/data-breach; figure cross-checked in press coverage of the 2026 report | global average breach cost US$4.99M, reported as a record |
| 7 | Google Cloud, "Sensitive Data Protection pricing." | https://cloud.google.com/sensitive-data-protection/pricing | live; content inspection US$3.00 per GiB from 1 GiB to 1 TiB per month, first 1 GiB free; content transformation US$2.00 per GiB; minimum of 1 KB billed per request for hybrid inspection. Because per-request minimums depend on batching, no total dollar figure is computed. |
| 8 | A. Riabi, M. Mahamdi, V. Mouilleron, D. Seddah, "Cloaked Classifiers: Pseudonymization Strategies on Sensitive Classification Tasks," Proc. 5th Workshop on Privacy in NLP, ACL, 2024, pp. 123-136. | https://aclanthology.org/2024.privatenlp-1.13/ (BibTeX) | match |
| 9 | Presidio documentation (presidio-analyzer 2.2.364 used). | https://presidio.dataprivacystack.org/ | page live |
| 10 | I. Pilán, P. Lison, L. Øvrelid, A. Papadopoulou, D. Sánchez, M. Batet, "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization," Computational Linguistics, vol. 48, no. 4, pp. 1053-1101, 2022. | https://aclanthology.org/2022.cl-4.19/ (BibTeX) | match |
| 11 | J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," Proc. NAACL-HLT 2019, pp. 4171-4186. | https://aclanthology.org/N19-1423/ (BibTeX) | match |
| 12 | sentence-transformers/all-MiniLM-L6-v2 model card, Apache-2.0. | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | live; license apache-2.0 |
| 13 | T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," Proc. EMNLP 2020: System Demonstrations, pp. 38-45. | https://aclanthology.org/2020.emnlp-demos.6/ (BibTeX) | match |
| 14 | F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, vol. 12, pp. 2825-2830, 2011. | https://jmlr.org/papers/v12/pedregosa11a.html | match |
| 15 | V. Sanh, L. Debut, J. Chaumond, T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019. | https://arxiv.org/abs/1910.01108 | match |
| 16 | Explosion, spaCy English pipelines: en_core_web_lg and en_core_web_trf 3.8.0, MIT license. | https://spacy.io/models/en | page live; both pipelines listed |
| 17 | W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, M. Zhou, "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers," Advances in NeurIPS 33, 2020. | https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html ; arXiv:2002.10957 | match |
| 18 | N. Reimers, I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," Proc. EMNLP-IJCNLP 2019, pp. 3982-3992. | https://aclanthology.org/D19-1410/ (BibTeX) | match |
| 19 | Qwen Team, "Qwen2.5-0.5B-Instruct," Hugging Face model card, Apache-2.0, 2024. | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct | live; 0.49B parameters (0.36B non-embedding); license apache-2.0 |
| 20 | T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," Neural Computation, vol. 10, no. 7, pp. 1895-1923, 1998. | Crossref record for doi:10.1162/089976698300017197 | match |
| 21 | B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, 1993, doi:10.1007/978-1-4899-4541-9. | Springer e-book record for the DOI (first published by Chapman & Hall) | title, authors, year match |
| 22 | S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65-70, 1979. | bibliographic records (JSTOR metadata) | match |
| 23 | E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference," Journal of the American Statistical Association, vol. 22, no. 158, pp. 209-212, 1927. | bibliographic records (publisher metadata) | match |
| 24 | dslim/bert-base-NER model card, MIT. | https://huggingface.co/dslim/bert-base-NER | live; license mit |
