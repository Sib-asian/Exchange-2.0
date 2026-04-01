# Worklog - Exchange-2.0 OCR Feature

---
Task ID: 1
Agent: Main Agent
Task: Risolvere definitivamente i problemi CI e OCR per la feature di upload screenshot

Work Log:
- Analizzata immagine dell'errore CI tramite VLM - problema: z-ai CLI non trovato nell'ambiente Streamlit
- Identificato che il CLI `z-ai` esiste in `/usr/local/bin/z-ai` come symlink a un file JS
- Modificato `src/ocr.py`:
  - Aggiunta funzione `_find_zai_command()` per ricerca robusta del CLI
  - Supporta: shutil.which, percorsi assoluti, esecuzione tramite bun/node
  - Aggiunta funzione `_check_zai_available()` helper
  - Aggiornato `extract_from_image_file()` per usare la nuova funzione di discovery
- Aggiornato `tests/test_ocr.py`:
  - Aggiunti test per `_find_zai_command` (TestFindZaiCommand)
  - Aggiunti test per `_check_zai_available` (TestCheckZaiAvailable)
  - Aggiornati test esistenti per usare mock di `_find_zai_command`
  - Aggiunto test per estrazione con bun runner
- Verificato che tutti i test passano: 276 passed
- Verificato coverage: 90.72% (sopra il 70% richiesto)
- Testato OCR con immagine reale: funziona correttamente
- Committato e pushato con messaggio descrittivo

Stage Summary:
- Commit: 3e806a2 - "fix: improve z-ai CLI discovery with multiple fallback paths"
- Branch: feature/ocr-screenshot-upload
- Coverage: 90.72% (276 tests passed)
- File modificati: src/ocr.py, tests/test_ocr.py

---
Task ID: 2
Agent: Main Agent
Task: Aggiungere supporto Google Gemini come fallback gratuito per OCR (OpenAI a pagamento)

Work Log:
- Analizzato screenshot errore 429 (quota exceeded OpenAI)
- Implementato sistema multi-backend con fallback automatico:
  - Backend primario: z-ai CLI (gratuito, locale)
  - Backend fallback: Google Gemini API (gratuito 1500 req/giorno)
- Nuove funzioni in `src/ocr.py`:
  - `_get_gemini_api_key()`: legge API key da env o Streamlit secrets
  - `_check_gemini_available()`: verifica se Gemini è configurato
  - `_extract_with_gemini()`: estrazione via Gemini Vision API
  - `_extract_with_zai_cli()`: refactoring estrazione CLI
- Aggiunto campo `backend_used` a ExtractedData per tracciare quale backend è stato usato
- Aggiornati test in `tests/test_ocr.py`:
  - TestGetGeminiApiKey (2 test)
  - TestCheckGeminiAvailable (2 test)
  - TestExtractWithGemini (2 test)
  - TestExtractWithZaiCli (2 test)
  - TestExtractFromImageFile con fallback (3 test)
- Coverage finale: 90.26% (283 test passati)

Stage Summary:
- Commit: c19e383 - "feat: add multi-backend OCR support (z-ai CLI + Google Gemini)"
- Branch: feature/ocr-screenshot-upload
- Coverage: 90.26% (283 tests passed)
- File modificati: src/ocr.py (+437 linee), tests/test_ocr.py
- Supporto gratuito per Streamlit Cloud tramite Gemini API
