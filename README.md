# HealthLens: A Multimodal RAG Framework for Simplifying and Comparing Medical Reports

HealthLens turns laboratory reports (digital PDF, scanned PDF, photo/scan image, text) into
**simple, grounded explanations**. It extracts the values deterministically, retrieves trusted
medical knowledge (RAG), and explains the report at three levels (beginner / intermediate /
technical) in English, Malayalam or Hindi. Reports from different dates can be compared and plotted.

> **Educational tool only.** HealthLens explains what tests measure and whether values are inside the
> laboratory's own range. It never diagnoses, prescribes, or claims a change is "good" or "bad".

## What is implemented

| # | Objective | Where |
|---|---|---|
| 1-3 | Accept PDF / image / scanned reports; OCR | `src/ingestion/` (pypdf, pypdfium2, Pillow, Tesseract) |
| 2, 4 | Extract test, value, unit, range -> structured JSON | `src/extraction/test_extractor.py` (digital and OCR row layouts) |
| - | Validate against the **lab's own** reference range | `src/validation/validator.py` |
| 5 | Trusted knowledge base (CBC, WBC differential, lipid, metabolic, liver, thyroid, urine, inflammatory markers) | `data/medical_knowledge/` |
| 6 | RAG retrieval (FAISS + sentence-transformers, metadata-aware re-ranking) | `src/retriever.py`, `src/vector_store.py` |
| 7 | LLM explanation (Gemini) or deterministic offline explainer | `src/generator.py`, `src/explainer.py` |
| 8 | Report summary + conversational Q&A | `RAGPipeline.explain_report`, `app.py`, `main.py --interactive` |
| 9 | Beginner / intermediate / technical levels | `src/explainer.py` |
| 10 | English / Malayalam / Hindi | `src/explainer.py` |
| 11 | Report comparison, trends, charts, plain-language description | `src/comparison/`, `describe_trends` |
| 12 | Evaluation (retrieval, relevance, grounding, readability, safety) | `evaluation/evaluate.py` |
| 13 | No diagnosis / no prescription | guardrail in `explainer.is_diagnosis_request`, prompts, tests |

## Architecture

```
Report (PDF / scan / image / text)
  -> file-type detection -> pypdf text  |  pypdfium2 + Tesseract OCR
  -> deterministic extraction (regex; PII removed) -> structured records
  -> validation vs. lab reference range  (NORMAL / LOW / HIGH / UNKNOWN)          [ground truth]
  -> per-test retrieval from FAISS knowledge base (RAG)                           [medical background]
  -> level + language + patient values + retrieved knowledge -> Gemini  (or offline explainer)
  -> explanation, summary, Q&A      |     comparison + trends + charts + description
```

Design principle: **numbers are never produced by the LLM.** Values, ranges, flags, percentage
changes and trends are computed in Python; the LLM (or the offline template) only explains them.

## Quick start

```bash
pip install -r requirements.txt
# OCR for scanned/image reports needs the Tesseract program:
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki    Ubuntu: sudo apt install tesseract-ocr
cp .env.example .env          # optional: add GEMINI_API_KEY for full LLM explanations + translation

streamlit run app.py          # web app: upload, explain, chat, compare
```

Without `GEMINI_API_KEY` everything still works using the offline explainer (full English;
Malayalam/Hindi use translated headings, statuses and disclaimers with English medical background text).
With a key, Gemini writes the whole explanation in the chosen level and language.

### Command line

```bash
python main.py                                   # document-processing demo (ingest -> extract -> validate -> compare -> trends)
python main.py --index --force                   # (re)build the knowledge-base index
python main.py --query "What does WBC measure?"  # general question
python main.py --explain data/sample_reports/sample_medical_report.pdf --level beginner --language ml
python main.py --explain report.png --question "What does my LDL mean?" --level technical
python main.py --interactive --report data/processed/sample_medical_report_validated.json
python main.py --status
python main.py --test                            # run all tests
```

In interactive mode use `/level beginner|intermediate|technical`, `/lang en|ml|hi`, `/report <file>`, `/quit`.

## Tests and evaluation

```bash
python -m pytest tests -q                         # unit + integration tests
python evaluation/evaluate.py                     # writes evaluation/RESULTS.md and results.json
EMBEDDING_BACKEND=sentence-transformers python evaluation/evaluate.py   # use this one for the final report
```

The evaluation measures retrieval hit@3, answer relevance, numeric grounding, report faithfulness,
Flesch readability per level, safety (refusals, disclaimer, no diagnostic assertions) and multilingual output.
The test questions are in `evaluation/evaluate.py` and are easy to extend.

## Project structure

```
app.py                      Streamlit web app
main.py                     CLI (document demo + RAG commands)
src/
  ingestion/                PDF + OCR loaders
  extraction/               regex extractor (digital layout + OCR single-line rows)
  validation/               lab-range validator
  comparison/               comparator, trend analyzer, Matplotlib visualizer
  integration/              zero-PII context interface
  explainer.py              levels, languages, summary, guardrail, prompts, offline explainer
  rag_pipeline.py           indexing, retrieval, explain_report()
  retriever.py / vector_store.py / embeddings.py / chunker.py / data_loader.py
  generator.py              Gemini + offline generation
data/medical_knowledge/     knowledge base (JSON + Markdown); each entry has a plain-language `simple_explanation`
evaluation/                 evaluation script and results
tests/                      pytest suites (Phases 1-3 + document-processing stages)
```

## Document-processing details

* **Reference ranges** always come from the report itself; knowledge-base ranges are labelled
  "general educational information" and never replace the laboratory's range.
* **Comparison** uses `absolute_change = current - previous` and
  `percentage_change = (current - previous) / |previous| * 100` (`None` if previous is 0). Directions are
  purely numerical (`INCREASE`, `DECREASE`, `UNCHANGED`, `ONLY_IN_PREVIOUS`, `ONLY_IN_CURRENT`,
  `UNIT_MISMATCH`); values with different units are never subtracted.
* **Privacy:** patient name, ID, age, doctor and contact details are removed before structured data is created.

## Extending the knowledge base

Add an entry to any JSON file in `data/medical_knowledge/` with `test_id`, `test_name`, `aliases`,
`category`, `description`, `what_it_measures`, `simple_explanation`, `standard_units`,
`reference_ranges`, and `clinical_significance`, then run `python main.py --index --force`.
If a test has no trustworthy entry, HealthLens says so instead of guessing.

## Limitations

* The extractor is rule-based: unusual layouts may need new patterns. OCR quality depends on scan quality.
* Offline Malayalam/Hindi output translates the structure, not the medical background text (use Gemini for that).
* The knowledge base is small and curated; qualitative urine/microscopy results are explained but not range-validated.
* Not a medical device and not a substitute for a clinician.

## Team

| Member | Scope |
|---|---|
| Person 1 | Document processing, extraction, validation, comparison, trends, visualization |
| Person 2 | Knowledge base, RAG, LLM generation, levels/languages, Q&A |
