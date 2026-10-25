# Local validation (2026-10-24)

The main Streamlit app, FastAPI backend, OpenAI generation, and both Pinecone
indexes were exercised with the PDF indexed in the isolated
`local-validation-20261024-v3` namespace. The default Pinecone namespace was
not changed.

- Offline tests: **18 passed** (`python -m pytest -q`).
- Live HTTP smoke test: Computer Science requirements, a follow-up cutoff
  question, and an unrelated question all returned valid answers and sources.
- Live Streamlit test: a Nursing (Obuasi Campus) cutoff question returned 12.
- RAGAS evaluation: **11 passed, 1 failed** out of 12, using the normal `/query`
  route and thresholds of 0.5 for context recall, faithfulness, and answer
  relevancy. The XML and JSON records are in ignored `.local/ragas-results.*`
  on the machine where the evaluation ran.

The failed fee question returned the documented US$100 international application
fee, correctly said the PDF does not specify a Ghanaian discount or a fee
comparison, and scored 1.0 for both context recall and faithfulness. RAGAS
assigned answer relevancy 0.0 because its noncommittal classifier treats a
qualified answer as noncommittal. Keep the failing test visible; its score alone
does not establish that the answer is wrong.

Manual review found accuracy gaps that the pass thresholds did not catch:

- The Computer Engineering comparison cited an exception for programme 10,
  although the PDF lists Computer Engineering as programme 11. The applicable
  exception is the one labelled for programmes 11, 12, and 13 (PDF page 30).
- The Nursing campus comparison reported only Obuasi's cutoff of 12 and omitted
  the main programme's cutoff of 07. It therefore could not establish the
  difference, despite passing the automated thresholds.
- The general mature applicant answer applied an agriculture-specific two-year
  experience requirement more broadly than the question warranted.

The API and UI work locally, including retrieval, follow-up handling, and
citations. The comparison and generalization examples above need further
retrieval and answer validation before relying on the assistant for admissions
decisions. The PDF's historical 2024/2025 cutoffs are not current guarantees.

On 2026-10-25, the services were restarted and the 18 offline tests, three live
API smoke questions, and the Streamlit-to-API cutoff question passed again.
