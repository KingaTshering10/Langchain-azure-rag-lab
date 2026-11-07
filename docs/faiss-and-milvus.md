# FAISS and Milvus: companion corrections

The preserved `faiss_vs_milvus_original.docx` is a supplied training note, not a verified feature matrix. This companion corrects its main persistence claim and avoids broad architecture assumptions.

| Topic | FAISS | Milvus |
| --- | --- | --- |
| Main role | Library for vector similarity search and clustering | Vector database for storing/searching embeddings with structured metadata |
| Persistence | Supports writing and reading indexes; application code manages associated document metadata | Provides database storage and management features |
| Use in this project | Exact cosine index with separately saved JSON documents | No Milvus implementation is included |

FAISS can persist indexes with `write_index` and `read_index`. Its being an embedded library does not mean it cannot save data. In this lab, the vector index and document metadata are separate files, and the notebook demonstrates a round trip.

Milvus has multiple deployment modes, including Lite, Standalone, and Distributed. Avoid treating all Milvus installations as a distributed server cluster or making blanket performance comparisons without a workload, hardware description, index configuration, and measurements.

This collection includes no FAISS-versus-Milvus benchmark. Choose an architecture using concrete storage, metadata, concurrency, operations, and scale requirements rather than the source note's simplified claims.

Primary references:

- [FAISS index I/O documentation](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning)
- [Milvus overview and deployment modes](https://milvus.io/docs/overview.md)
