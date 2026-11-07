# Preparation and validation notes

## Source inspection

Reviewed all 16 substantive files in `8langchain(1).zip`: six notebooks, two `.env` files, two PDFs, one JSON profile, one DOCX comparison, one dependency file, one Streamlit script, and two cached-index files. macOS resource metadata was excluded. The original notebooks' sources and output counts were inspected, and PDF text and DOCX paragraph/table text were extracted. The cached pickle was not deserialized.

The HR PDF has ten pages and identifies the OpenGov Foundation with a January 8, 2018 update date. The public OpenGov Foundation manual was checked for matching source organization/date. The New York PDF has one page and identifies Margie's Travel. Both PDF files are retained unchanged. The comparison DOCX's OOXML archive passed an integrity check; it is retained unchanged with a corrected Markdown companion. No Office/PDF layout changes were made or visually validated.

## Changes made

1. Removed the two source `.env` files and all saved notebook outputs, execution counts, and original notebook metadata. Added a root `.env.example` and centralized settings without printing keys. Existing credentials must be rotated separately if real.
2. Rebuilt and renamed the six notebooks into a learning sequence. Removed inaccurate open-source-model wording, obsolete chain-class headings, empty cells, and suppressed warning blocks.
3. Made metadata lookups optional rather than assuming nested token-usage fields always exist. Used `StrOutputParser` and `JsonOutputParser` to pass plain text or parsed objects between runnable stages.
4. Resolved loan/insurance naming and input mismatches in the original workflow. The insurance example uses a catalog of illustrative names/IDs, deterministic category lookup after model parsing, and a clarification-focused follow-up. It does not invent product benefits, prices, or approval rules. The router-chain topic was in the source outline but had no working implementation; no router feature is claimed.
5. Excluded the original named financial profile. Added a newly constructed synthetic customer fixture with no name, location, age, marital status, or dependents. Reframed the template exercise as factual summarization because the source provided no validated formula for the requested credit-risk score or approval decision.
6. Switched from deprecated community embedding imports to `langchain_huggingface.HuggingFaceEmbeddings`. Replaced DocArray with LangChain Core's in-memory store. Dependency files no longer contain contradictory Streamlit pins, NumPy 1.25.2, or unused DocArray/community-wrapper packages.
7. Added overlapping PDF chunks and explicit source/page metadata. Similarity filtering is applied explicitly rather than assuming a `score_threshold` argument enables filtering under plain similarity search. Scores are described as cosine similarity, not confidence.
8. Excluded the supplied `index.faiss` and `index.pkl`. The app builds from the packaged PDF and caches by source hash/model. The persistence exercise saves a native FAISS index plus JSON metadata, checks model/dimension/metric/checksum, and uses no pickle loading. Native index files should still be loaded only from a trusted local source; a checksum is not authentication.
9. Added missing-configuration guidance, source display, error handling, and clear-history controls to the Streamlit app. Chat history is a display log; each question remains a standalone retrieval query.
10. Added the README, repository description, corrected FAISS/Milvus note, byte-preservation manifest, and offline regression checks.

## Validation performed

- All six rebuilt notebooks passed `nbformat` validation and Python syntax parsing. Code files also parsed successfully. Notebook outputs and execution counts are empty.
- Six offline regression tests passed using actual FAISS and LangChain Core. Deterministic test embeddings exercised ranking, score cutoffs, large-k limits, invalid input handling, save/load, model mismatch, checksum failure, and source metadata.
- The actual LCEL workflow notebook was executed using fake `AIMessage` responses, exercising the sequential, parallel, parsing, and named-output stages.
- PDF chunking was exercised on both original PDFs. Document content and page references were checked against the extracted text; source files were not modified.
- Streamlit AppTest passed for missing configuration, a mock generated answer using actual FAISS retrieval, displayed source records, and clear history. No real Azure requests or model downloads were used by these checks.
- Tested runtime: OpenAI SDK 3.26.0, LangChain OpenAI 1.6.7, LangChain Core 1.7.6, FAISS CPU 1.15.1, NumPy 2.5.4, Streamlit 1.67.0, and nbformat 5.11.1. Requirements use compatibility ranges and selected pins rather than a complete lockfile.
- Retained binary documents were byte-compared against the source and recorded with SHA-256 hashes in `source-assets.json`. Finished files were checked for original credential values and common credential patterns. No real `.env`, cached source index, or pickle file is packaged.

## Not measured

No live Azure response, real BGE embedding inference, deployed application session, retrieval-quality benchmark, financial model, or quantitative model comparison was performed. Test embeddings validate the mechanics of retrieval rather than semantic accuracy. Prompt instructions do not establish faithful citations or reliable abstention.

## Attribution and source limits

The sample PDFs are historical training context. The HR policy belongs to its stated source organization; the travel document contains historical population text and sample hotel descriptions. Neither is a statement of current policy or travel availability. The original DOCX includes unsupported generalizations and is accompanied by corrections. No complete collection-wide license or author manifest was supplied; packaging adds no ownership claim over original materials.
