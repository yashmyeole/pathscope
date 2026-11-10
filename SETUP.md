# Accounts and keys

Everything runs on free tiers. Reproducing the report needs none of this (`uv run pathscope report`
works from a clean clone); the steps below are for re-running the pipeline or hosting the demo.

## Qdrant Cloud

1. Sign up at https://cloud.qdrant.io (no card needed) and create a cluster on the free tier.
2. Copy the cluster URL (ends in `:6333`) and create an API key.
3. Put both in `.env`:
   ```bash
   cp .env.example .env     # QDRANT_URL, QDRANT_API_KEY
   uv sync
   uv run pathscope ping    # {"collections": [...]}
   ```

Free clusters suspend after 7 idle days and are deleted after 28.

## Keep-alive action

`.github/workflows/qdrant-keepalive.yml` pings the cluster every 3 days. It needs two repository
secrets (Settings → Secrets and variables → Actions):

| Name | Value |
|---|---|
| `QDRANT_URL` | the cluster URL, with `:6333` |
| `QDRANT_API_KEY` | the API key |

Run it once by hand from the Actions tab to check it. GitHub pauses scheduled workflows after 60
days without a commit.

## LLM keys

Only needed to re-run claim extraction, checking and generation. All responses are cached in
`data/llm_cache/`, so a run that hits a daily limit can be restarted the next day.

- Groq: https://console.groq.com → `GROQ_API_KEY`
- Gemini: https://aistudio.google.com/apikey → `GEMINI_API_KEY`

## Hugging Face

PathVQA and PLIP are public, so a token is optional. An `HF_TOKEN` in `.env` only raises the
download rate limit.

## Vercel

1. Sign in at https://vercel.com with GitHub (Hobby plan) and import the repository.
2. Set **Root Directory** to `app`. Next.js is detected automatically.
3. Add `QDRANT_URL` and `QDRANT_API_KEY` as environment variables.
4. Deploy. Every push to `main` redeploys.

If the cluster can't be reached, the demo still works with the saved neighbours and says so.
