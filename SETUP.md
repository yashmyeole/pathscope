# One-time setup (Phase 0 manual steps)

These are the account/credential steps nobody can do for you. ~30 minutes total.
Tick them off in order; the keep-alive action only works after steps 1–3.

## 1. GitHub repo (5 min)

```bash
cd ~/Desktop/Projects/UIC/SEM3/VectorDB/VectorDB_CS594
git init -b main                       # already done if you see a .git/ folder
gh repo create pathscope --public --source=. --remote=origin   # or create on github.com and `git remote add origin ...`
git add -A && git commit -m "Phase 0: repo scaffold, keep-alive action, PathVQA stats"
git push -u origin main
```

Public repo is intentional — it is the one that goes on the resume.

## 2. Qdrant Cloud free cluster (5 min)

1. https://cloud.qdrant.io → sign up (GitHub login is fine, no card).
2. **Create cluster** → *Free* tier → pick a US region (`us-east4` GCP is fine) → name `pathscope`.
3. When it is running, open the cluster page and copy:
   - **Cluster URL** — looks like `https://xxxxxxxx-....us-east4-0.gcp.cloud.qdrant.io:6333`
   - **API key** — generate one under *API Keys*, copy it once (it is not shown again).
4. Locally:
   ```bash
   cp .env.example .env     # paste QDRANT_URL and QDRANT_API_KEY
   uv sync
   uv run pathscope ping    # expect: {"collections": []}
   ```

> Free clusters **suspend after 7 idle days and are deleted after 28.** Step 3 prevents that.

## 3. GitHub secrets for the keep-alive action (3 min)

Repo → *Settings* → *Secrets and variables* → *Actions* → *New repository secret*:

| Name | Value |
|---|---|
| `QDRANT_URL` | the cluster URL from step 2 (with `:6333`) |
| `QDRANT_API_KEY` | the API key from step 2 |

Then *Actions* tab → **Qdrant keep-alive** → *Run workflow* to test it. A green run that prints
`{"ts": ..., "ok": true, "collections": [...]}` is the finish line for Phase 0 infrastructure.

Scheduled workflows on GitHub are paused automatically after 60 days with no repo activity —
any commit resets that clock, which will happen naturally during the semester. Put a reminder for
the winter break.

## 4. HuggingFace account (5 min, long lead time)

1. https://huggingface.co/settings/account → change the **primary email** to your `@uic.edu` address
   and verify it. Gated pathology models reject gmail addresses.
2. Request access now (takes days to weeks, costs nothing, not needed for v1):
   - CONCH: https://huggingface.co/MahmoodLab/CONCH
   - UNI:   https://huggingface.co/MahmoodLab/UNI
3. Create a read token at https://huggingface.co/settings/tokens → `HF_TOKEN` in `.env`.
   PLIP (`vinid/plip`) is ungated, so nothing in Phase 2 blocks on this.

## 5. Kaggle (optional now)

Phase 2 was run on CPU (PLIP is small: ~11.6k images in ~20 min), so Kaggle is no longer needed for
v1. Verify a phone at https://www.kaggle.com only if you want a free GPU later (Phase 5 upgrades).

## 6. Ask about pathologist access (longest lead time of anything)

Email Ron / the professor: "Is there anyone in the UIC pathology department who would spend ~2 hours
labelling a 50-item subset later in the semester?" Not needed for v1, but it unblocks the strongest
upgrade in Part 5 of the guide.

## 7. API keys for Phase 4–5 (do when you get there)

- Groq: https://console.groq.com → API key → `GROQ_API_KEY`
- Gemini: https://aistudio.google.com/apikey → `GEMINI_API_KEY`

---

**Phase 0 finish line:** `uv run pathscope ping` works locally, the GitHub Action is green, and
`data/stats/pathvqa_stats.md` + the contact sheets exist and you have actually looked at them.
