# Contributing to DocSage

Thank you for helping improve DocSage. This document covers local development, testing expectations, and what we look for in pull requests.

Application code lives under `docsage/`. The project root `README.md` describes architecture and runtime behavior.

## Local development

### Prerequisites

- Python 3.11 or newer (recommended)
- Node.js 18 or newer (Node 20 is a good default)
- `pip` and `npm`
- Optional: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your machine if you work on PDFs that require OCR

### Backend

From the repository root:

```bash
cd docsage/backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive API docs: `http://localhost:8000/api/docs`
- Health check: `http://localhost:8000/api/v1/health`

Create a `.env` file in `docsage/backend/` if you need to override defaults. Settings are loaded from `core/config.py` (see `Settings` and `env_file`).

### Frontend

```bash
cd docsage/frontend
npm install
npm run dev
```

The dev server is typically at `http://localhost:5173`. Point the UI at your API if needed:

```bash
export VITE_API_URL="http://localhost:8000/api/v1"
```

### Docker

Compose and Dockerfiles live under `docsage/docker/`. See `README.md` for the current compose versus filename caveats before relying on container workflows for development.

### Data and caches

Runtime data (uploaded files, indices, Hugging Face cache) is under `docsage/backend/data/`. Do not commit large or machine-specific artifacts; prefer local `.gitignore` hygiene and documenting required env vars instead.

## Code style and conventions

- **Scope:** Keep changes focused on the issue or feature. Avoid unrelated refactors, formatting-only sweeps, or new dependencies unless necessary.
- **Consistency:** Match existing naming, imports, and patterns in the files you touch.
- **Python:** Prefer clear functions and typed boundaries where the rest of the module already uses types. Use `structlog` patterns in `core/logging.py` where logging is appropriate.
- **JavaScript/React:** Keep components readable; follow existing component structure under `docsage/frontend/src/`.
- **Configuration:** Prefer `core/config.py` and environment variables over hard-coded hosts, paths, or secrets.
- **Security:** Do not commit secrets, API keys, or production credentials. Use placeholders in examples.

## Testing standards

### Backend

- Tests live in `docsage/backend/tests/`.
- Run the suite from `docsage/backend` with:

  ```bash
  pytest -q
  ```

- For coverage while iterating:

  ```bash
  pytest --cov=. --cov-report=term-missing
  ```

  (Adjust paths if you only want `api`, `core`, `models`, `utils`.)

**When to add or update tests**

- **API changes:** Extend or add tests in `tests/test_api.py` (or new modules under `tests/`) for new routes, status codes, and validation.
- **Core logic:** Add unit tests for chunking, retrieval edge cases, or pipeline behavior when behavior changes are non-trivial.
- **Mocks:** Prefer mocking heavy dependencies (models, filesystem, external HTTP) in tests so CI and local runs stay fast and deterministic.

**What we expect**

- New behavior should include tests when the change is user-visible or fixes a bug (regression test).
- Existing tests must pass before a PR is merged.

### Frontend

There is no shared frontend test runner in the minimal Vite setup today. If you add tests (for example Vitest or React Testing Library), document the commands in this file and keep scripts in `package.json` so `npm test` (or the chosen script) is obvious for contributors.

## Pull request checklist

Use this list before you open or mark a PR ready for review.

- [ ] **Purpose:** The PR description states what changed and why, with a link to an issue if one exists.
- [ ] **Scope:** Changes are limited to what is needed; unrelated files are not modified.
- [ ] **Tests:** Backend tests pass (`pytest` from `docsage/backend`). New or updated tests cover the change where appropriate.
- [ ] **Manual check:** You ran the app locally (backend and, if UI changed, frontend) and verified the happy path.
- [ ] **Config / docs:** `README.md` or this file is updated if setup, env vars, or scripts changed.
- [ ] **Breaking changes:** Any API or behavior change is called out in the PR description with migration notes if needed.
- [ ] **Artifacts:** No accidental commits of secrets, large binaries, or local `venv` / `node_modules` / model cache paths unless the project explicitly requires them (it should not).

## Review expectations

Maintainers may ask for smaller follow-up PRs, additional tests, or naming clarifications. Keeping PRs small and well-described speeds up review.

If you are unsure whether a change needs tests or documentation, err on the side of a short note in the PR and a minimal test or README touch so the next contributor understands the intent.
