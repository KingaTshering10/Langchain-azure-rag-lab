# LangChain Azure RAG Lab

Build from a first Azure OpenAI call to a document question-answering app using **LangChain, LCEL, local embeddings, and FAISS**.

This hands-on learning collection includes **six notebooks**, a **Streamlit PDF Q&A app**, reusable Python helpers, sample documents, and offline regression checks. Follow the notebooks in order or start the app to explore the supplied New York training brochure.

## What you will learn

- Connect a deployed Azure OpenAI chat model through LangChain.
- Inspect message objects and optional token-usage metadata.
- Build prompt templates and parse JSON or plain-text outputs.
- Compose sequential and parallel LCEL workflows with named results.
- Split PDFs into chunks and retrieve context with source/page metadata.
- Compare an in-memory vector store with a FAISS cosine index.
- Save a FAISS index with JSON document metadata and reload it locally.
- Display generated answers, source passages, and similarity scores in Streamlit.

## Learning path

| Notebook | Topic | Main concepts |
| --- | --- | --- |
| [01 · Azure basics](notebooks/01_azure_basics.ipynb) | First model request | `AzureChatOpenAI`, `invoke`, message content |
| [02 · Response metadata](notebooks/02_response_metadata.ipynb) | Understand the response | Provider metadata, optional usage fields |
| [03 · Prompt templates](notebooks/03_prompt_templates.ipynb) | Structured practice data | `ChatPromptTemplate`, input variables, `StrOutputParser` |
| [04 · LCEL workflows](notebooks/04_lcel_workflows.ipynb) | Compose multiple steps | `JsonOutputParser`, `RunnableLambda`, `RunnableParallel`, `assign` |
| [05 · HR PDF RAG](notebooks/05_hr_pdf_rag.ipynb) | Retrieve from a policy document | `InMemoryVectorStore`, chunking, explicit score filtering, grounded answers |
| [06 · FAISS RAG](notebooks/06_faiss_rag.ipynb) | Search and persist a local index | Normalized embeddings, cosine search, native index + JSON metadata |

The examples use LCEL runnable compositions. Legacy `LLMChain`, `SequentialChain`, and router-chain implementations are not required. The original outline mentioned router chains, but did not supply a working router example.

## Quick start on your Mac

Use **Python 3.11** as the common starting point. Open the extracted or cloned repository folder in VS Code, then run these commands from its root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your own Azure resource details:

```dotenv
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=replace-with-your-key
AZURE_OPENAI_DEPLOYMENT=your-chat-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21
```

Use the **deployment name you created**, which can differ from the model name. These examples retain the dated `AzureChatOpenAI` API approach; select an API version supported by your resource. LangChain also documents Azure's newer v1 connection approach in the references below.

Select the `.venv` interpreter as the notebook kernel in VS Code, or launch JupyterLab:

```bash
jupyter lab
```

Run notebook cells from top to bottom. Root discovery works when the kernel's working directory is inside the repository.

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` to activate the environment and `Copy-Item .env.example .env` to create the configuration file.

## Run the PDF Q&A app

Install the additional retrieval and app dependencies:

```bash
python -m pip install -r requirements-rag.txt
streamlit run app.py
```

Try: **“Which hotels are listed in the New York brochure?”**

On the first question, the app reads the included PDF, splits it into overlapping chunks, downloads the embedding model if needed, and creates a FAISS index. It caches that index in memory for subsequent questions. The cache includes the PDF content hash and embedding model name, so changed inputs rebuild it.

The app retrieves passages, sends the question and retrieved text to Azure, and shows the answer with source passages. Adjust the number of passages and optionally apply a cosine-similarity cutoff in the sidebar. The cutoff is **not calibrated confidence**. Each question is processed independently; the visible chat history is not added to later requests. Use **Clear conversation** to reset the display.

Model downloads require internet and local disk space. BGE embeddings run locally on CPU; answer generation uses your Azure deployment and may incur usage charges. Retrieved PDF text is sent to that deployment.

## Retrieval and persistence

The default embedding model is [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), an English embedding model with 384-dimensional outputs and a 512-token input limit. PDFs are split into 900-character chunks with 150 characters of overlap to reduce full-page truncation; token length still depends on text content.

The FAISS helper uses `IndexFlatIP` with normalized document and query vectors, giving exact cosine-similarity search. This example uses FAISS directly behind a small helper, then connects retrieval to LangChain prompt/model/parser stages. It does not depend on the older community FAISS wrapper or DocArray.

Notebook 06 can save a locally created index under `generated/new_york_index/`:

| File | Contents |
| --- | --- |
| `index.faiss` | Native FAISS vectors/index |
| `documents.json` | Text, metadata, model name, dimensions, and index checksum |

The loader checks the model name, dimensions, metric, and checksum. Load only index files you created locally: a checksum detects mismatched files but does not authenticate their source. Generated indexes are ignored by Git. The original prebuilt index and pickle metadata are excluded from this package.

## Included data

| Asset | Purpose and source context |
| --- | --- |
| [HR policy manual](assets/hr_policy_manual.pdf) | Ten-page OpenGov Foundation manual dated January 8, 2018; historical retrieval practice |
| [New York brochure](assets/new_york_brochure.pdf) | One-page Margie's Travel training brochure; sample hotel descriptions and historical population text |
| [Customer example](assets/customer_example.json) | Newly constructed, synthetic practice fixture for prompt templates; currency is unspecified |
| [Original FAISS/Milvus comparison](docs/faiss_vs_milvus_original.docx) | Preserved source note with known inaccuracies; read the [corrected companion](docs/faiss-and-milvus.md) |

The HR PDF describes its source organization's historical policy, rather than your employer's current policy. The travel brochure is sample context rather than a current booking directory. The financial template exercise summarizes supplied facts; it does not generate a lending approval, an eligibility decision, or a validated credit-risk score.

## Local verification

After installing `requirements-rag.txt`, run from the repository root:

```bash
python -m unittest discover -s tests -v
```

Six offline tests cover cosine ranking/filtering, index save/load and mismatch checks, PDF chunks and metadata, missing-context handling, configuration, and execution of the actual LCEL workflow notebook with fake model responses. FAISS operations use deterministic test embeddings and require no model download or Azure credentials.

All six notebooks passed syntax and format validation. Streamlit AppTest checks passed for missing configuration, a mock answer backed by actual FAISS retrieval, displayed sources, and clearing history. **Live Azure generation and real BGE embedding inference were not run.** No new accuracy, latency, or cost measurements are claimed. See [review notes](docs/review-notes.md) for tested versions and preparation details.

## Project files

- `notebooks/` — the six learning exercises
- `assets/` — sample PDFs and synthetic JSON fixture
- `lab/config.py` — root configuration and Azure model factory
- `lab/retrieval.py` — PDF chunking, embeddings, FAISS search, persistence, and answer generation
- `app.py` — Streamlit document Q&A interface
- `tests/` — offline regression checks
- `docs/` — review notes, attribution, and corrected comparison notes
- `description.txt` — short GitHub repository description

## Limits and reuse

Retrieval can return nearby passages even when a document cannot answer the question. A grounding prompt and citations do not guarantee factual answers; inspect the source passages. JSON parsing verifies syntax rather than the truth of extracted fields. The collection implements no production access controls, underwriting, router agent, or evaluation benchmark.

Real `.env` files and notebook outputs were removed. If any original keys are real, rotate them; removing them from this package does not revoke them. Never commit your local `.env`.

Prepared from the supplied `8langchain(1).zip` training archive. The two PDFs and original comparison DOCX retain their source bytes. The named financial profile and the old cached index are excluded. No complete collection-wide license or authorship record was supplied, so no blanket license has been assigned to third-party materials.

## References

- [LangChain: AzureChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/azure_chat_openai)
- [LangChain: HuggingFaceEmbeddings](https://reference.langchain.com/python/langchain-huggingface/embeddings/huggingface/HuggingFaceEmbeddings)
- [FAISS: index types and normalized inner product](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS: index I/O](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning)
- [OpenGov Foundation: HR manual source](https://github.com/opengovfoundation/hr-manual/blob/master/markdown/manual.md)
