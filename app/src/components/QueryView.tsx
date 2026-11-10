"use client";

import { useEffect, useState } from "react";

import type { Claim, Condition, Neighbour, QueryRecord } from "@/lib/shared";
import { imageUrl, sourceName } from "@/lib/shared";

type Live =
  | { state: "loading" }
  | { state: "live"; ms: number; total: number; same: number; ordered: boolean }
  | { state: "saved"; reason: string };

const tabs: { key: Condition; label: string; note: string }[] = [
  {
    key: "full",
    label: "Image + these captions",
    note: "The full system: Gemini saw the image and the five captions above.",
  },
  {
    key: "blind",
    label: "Captions only",
    note: "Control: the same five captions and no image. The gap to the full system is what the image adds.",
  },
  {
    key: "random",
    label: "Image + random captions",
    note: "Control: the image with five captions picked at random from the corpus. The gap to the full system is what the right captions add.",
  },
];

const markOf = { supported: "✓", contradicted: "✗", silent: "–" };
const titleOf = {
  supported: "Supported by the image's PathVQA answers",
  contradicted: "Contradicted by the image's PathVQA answers",
  silent: "PathVQA says nothing about this, so it can't be checked",
};
const tierName = { A: "visual", B: "diagnostic", C: "case-specific" };

function hideBroken(e: React.SyntheticEvent<HTMLImageElement>) {
  e.currentTarget.style.visibility = "hidden";
}

export default function QueryView({ rec }: { rec: QueryRecord }) {
  const [live, setLive] = useState<Live>({ state: "loading" });
  const [tab, setTab] = useState<Condition>("full");
  const [lit, setLit] = useState<number | null>(null);

  useEffect(() => {
    let gone = false;
    fetch(`/api/neighbours/${rec.id}`)
      .then(async (r) => {
        const body = await r.json();
        if (gone) return;
        if (!r.ok || !body.neighbours?.length) {
          setLive({ state: "saved", reason: body.error ?? `HTTP ${r.status}` });
          return;
        }
        // The page keeps the saved list, since the descriptions were written from it. The live
        // search is shown alongside to prove the vector DB returns the same thing.
        const got: string[] = body.neighbours.map((n: Neighbour) => n.id);
        const saved = rec.neighbours.map((n) => n.id);
        setLive({
          state: "live",
          ms: body.qdrantMs,
          total: body.totalMs,
          same: saved.filter((id) => got.includes(id)).length,
          ordered: got.join() === saved.join(),
        });
      })
      .catch(() => !gone && setLive({ state: "saved", reason: "network error" }));
    return () => {
      gone = true;
    };
  }, [rec.id, rec.neighbours]);

  const open = rec.facts.filter((f) => f.a.toLowerCase() !== "yes");
  const yes = rec.facts.filter((f) => f.a.toLowerCase() === "yes");
  const gen = rec.conditions[tab];
  const linksToNeighbours = tab !== "random";

  return (
    <>
      <div className="layout">
        <section>
          <div className="panel">
            <img className="query-img" src={imageUrl(rec.id)} alt="Query image" />
            <div className="tags">
              <span className="tag">{rec.kind}</span>
              <span className="tag">PathVQA test</span>
              {rec.labelMatch && (
                <span
                  className="tag"
                  title="Some corpus image has exactly the same short label, usually another panel of the same textbook figure."
                >
                  label twin in corpus
                </span>
              )}
            </div>
          </div>
          <div className="panel" style={{ marginTop: 14 }}>
            <h3>What PathVQA records about this image</h3>
            <p className="note">This is the evidence each claim below is checked against.</p>
            <ul className="facts">
              {open.map((f, i) => (
                <li key={i}>
                  <span className="muted">{f.q}</span> {f.a}
                </li>
              ))}
            </ul>
            {yes.length > 0 && (
              <details>
                <summary>{yes.length} yes/no questions answered “yes”</summary>
                <ul className="facts">
                  {yes.map((f, i) => (
                    <li key={i}>{f.q}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="nb-head">
            <h3>Five nearest images in the corpus</h3>
            <span className="live">
              {live.state === "loading" && "Asking Qdrant…"}
              {live.state === "live" && (
                <>
                  <b>●</b> live from Qdrant, {live.ms} ms search ({live.total} ms round trip)
                  {live.ordered
                    ? ", same five as below"
                    : live.same === 5
                      ? ", same five in a different order (tied scores)"
                      : `; ${live.same} of these 5 come back (tied scores). Below is the list the descriptions used`}
                </>
              )}
              {live.state === "saved" && "Saved results (vector search unreachable right now)"}
            </span>
          </div>
          <p className="note">Ranked by PLIP cosine similarity. Their captions are what the description is built from.</p>
          {rec.neighbours.map((n) => (
            <div key={n.rank} className={`nb${linksToNeighbours && lit === n.rank ? " hl" : ""}`}>
              <img src={imageUrl(n.id)} alt="" loading="lazy" onError={hideBroken} />
              <div>
                <div className="meta">
                  #{n.rank} · similarity {n.score.toFixed(3)} · {sourceName[n.source] ?? n.source} · {n.kind}
                </div>
                <div className="cap">{n.caption || <span className="muted">(no caption)</span>}</div>
              </div>
            </div>
          ))}
        </section>
      </div>

      <section className="panel" style={{ marginTop: 24 }}>
        <h3>Generated description</h3>
        <div className="tabs" role="tablist">
          {tabs.map((t) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={tab === t.key}
              onClick={() => {
                setTab(t.key);
                setLit(null);
              }}
              disabled={!rec.conditions[t.key]}
            >
              {t.label}
            </button>
          ))}
        </div>
        <p className="note">{tabs.find((t) => t.key === tab)?.note}</p>
        {gen ? (
          <>
            <p className="desc">{gen.description}</p>
            <Tally claims={gen.claims} />
            <ul className="claims">
              {gen.claims.map((c, i) => (
                <ClaimRow
                  key={i}
                  c={c}
                  link={linksToNeighbours}
                  onEnter={() => setLit(c.from)}
                  onLeave={() => setLit(null)}
                />
              ))}
            </ul>
            {tab === "random" && gen.captions && (
              <>
                <p className="note" style={{ marginTop: 14 }}>
                  The random captions it was given:
                </p>
                <ol className="randcaps">
                  {gen.captions.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ol>
              </>
            )}
          </>
        ) : (
          <p className="muted">No description for this condition.</p>
        )}
        <div className="legend">
          <span className="A">Tier A, visual</span>
          <span className="B">Tier B, diagnostic</span>
          <span className="C">Tier C, case-specific</span>
          <span>✓ supported · ✗ contradicted · – can&apos;t be checked</span>
        </div>
      </section>
    </>
  );
}

function Tally({ claims }: { claims: Claim[] }) {
  const n = (l: Claim["label"]) => claims.filter((c) => c.label === l).length;
  return (
    <div className="tally">
      {claims.length} claims: {n("supported")} supported, {n("contradicted")} contradicted,{" "}
      {n("silent")} can&apos;t be checked
    </div>
  );
}

function ClaimRow(props: { c: Claim; link: boolean; onEnter: () => void; onLeave: () => void }) {
  const { c, link } = props;
  const linked = link && c.from != null;
  const where = c.from == null ? null : link ? `#${c.from}` : `random caption ${c.from}`;
  return (
    <li
      className={`claim ${c.tier}${linked ? " linked" : ""}`}
      onMouseEnter={props.onEnter}
      onMouseLeave={props.onLeave}
      onFocus={props.onEnter}
      onBlur={props.onLeave}
      tabIndex={linked ? 0 : undefined}
    >
      <span className={`mark ${c.label}`} title={titleOf[c.label]}>
        {markOf[c.label]}
      </span>
      <span>{c.text}</span>
      <span className="src">
        {tierName[c.tier]}
        {where && ` · ${c.copied ? "copied from" : "from"} ${where}`}
      </span>
    </li>
  );
}
