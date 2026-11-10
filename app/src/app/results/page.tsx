import type { Metadata } from "next";

import { results } from "@/lib/data";

export const metadata: Metadata = { title: "PathScope · Results" };

const f2 = (x: number | null | undefined) =>
  x == null ? "–" : x < 0 ? `−${Math.abs(x).toFixed(2)}` : x.toFixed(2);
const ci = (c: number[] | null | undefined) => (c ? `${f2(c[0])} to ${f2(c[1])}` : "–");
const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const signed = (x: number) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`;

const tierRows = [
  { k: "A", name: "A, visual (organ, stain, tissue, cells)" },
  { k: "B", name: "B, diagnostic (disease, process)" },
  { k: "C", name: "C, case-specific (patient, grade, size)" },
] as const;

const condRows = [
  { k: "full", name: "Image + retrieved captions" },
  { k: "random", name: "Image + random captions" },
  { k: "blind", name: "Retrieved captions, no image" },
] as const;

export default function Results() {
  const p4 = results.phase4;
  const p5 = results.phase5;
  const twin = p5.byLabelMatch;

  return (
    <div className="prose" style={{ maxWidth: 900 }}>
      <h1>Results</h1>
      <p className="lead">
        The transferable claim rate (TCR) is the share of checkable claims that are true of the new
        image: supported / (supported + contradicted). Claims PathVQA says nothing about are never
        counted as wrong. Intervals are 95% bootstrap CIs over images.
      </p>

      <h2>1. Raw neighbour captions</h2>
      <p>
        Every caption of the 5 nearest corpus images, for all {p4.queries} test images, split into{" "}
        {p4.overall.claims.toLocaleString()} claim checks.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Tier</th>
              <th>claims</th>
              <th>checkable</th>
              <th>TCR</th>
              <th>95% CI</th>
            </tr>
          </thead>
          <tbody>
            {tierRows.map((t) => {
              const r = p4.byTier[t.k];
              return (
                <tr key={t.k}>
                  <td>{t.name}</td>
                  <td className="n">{r.claims.toLocaleString()}</td>
                  <td className="n">{pct(r.decided)}</td>
                  <td className="n">{f2(r.tcr)}</td>
                  <td className="n">{ci(r.ci95)}</td>
                </tr>
              );
            })}
            <tr>
              <td>
                <b>All</b>
              </td>
              <td className="n">{p4.overall.claims.toLocaleString()}</td>
              <td className="n">{pct(p4.overall.decided)}</td>
              <td className="n">
                <b>{f2(p4.overall.tcr)}</b>
              </td>
              <td className="n">{ci(p4.overall.ci95)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Case-specific claims can&apos;t be measured here: PathVQA records almost nothing about the
        patient, so 95% of them can&apos;t be checked. Visual and diagnostic claims transfer at about
        the same rate.
      </p>

      <h3 style={{ marginTop: 22 }}>Rank matters more than the similarity score</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Neighbour rank</th>
              {[1, 2, 3, 4, 5].map((r) => (
                <th key={r}>{r}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(["A", "B"] as const).map((t) => (
              <tr key={t}>
                <td>TCR, tier {t}</td>
                {p4.byRank[t].map((x, i) => (
                  <td key={i} className="n">
                    {f2(x)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        The top neighbour&apos;s diagnosis holds most of the time. From rank 3 on it&apos;s worse
        than a coin flip. Across images, the raw cosine score barely predicts transfer (Spearman ρ ={" "}
        {f2(p4.spearman?.rho)}).
      </p>

      <h2>2. Generated descriptions and two controls</h2>
      <p>
        {p5.queries} test images, each described three ways by Gemini, then scored with the same
        claim checks.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Condition</th>
              <th>TCR</th>
              <th>95% CI</th>
              <th>right / description</th>
              <th>wrong / description</th>
              <th>copied</th>
            </tr>
          </thead>
          <tbody>
            {condRows.map((c) => {
              const r = p5.conditions[c.k];
              return (
                <tr key={c.k}>
                  <td>{c.name}</td>
                  <td className="n">
                    <b>{f2(r.tcr)}</b>
                  </td>
                  <td className="n">{ci(r.ci95)}</td>
                  <td className="n">{f2(r.supportedPerDescription)}</td>
                  <td className="n">{f2(r.contradictedPerDescription)}</td>
                  <td className="n">{pct(r.copyRate)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p>
        Seeing the image adds <b>{signed(p5.multimodalGain.gain)}</b> TCR (CI{" "}
        {ci(p5.multimodalGain.ci95)}). Getting the right captions instead of random ones adds{" "}
        <b>{signed(p5.retrievalGain.gain)}</b> (CI {ci(p5.retrievalGain.ci95)}). Without the image,
        the model blends five captions that disagree and commits to one guess, so it makes about
        twice as many wrong claims.
      </p>

      {twin && (
        <>
          <h3 style={{ marginTop: 22 }}>The image only matters when there is no label twin</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th>images</th>
                  <th>image + captions</th>
                  <th>captions only</th>
                  <th>image gain</th>
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ["true", "Corpus has an image with the same label"],
                    ["false", "No label twin"],
                  ] as const
                ).map(([k, name]) => {
                  const r = twin[k];
                  return (
                    <tr key={k}>
                      <td>{name}</td>
                      <td className="n">{r.queries}</td>
                      <td className="n">{f2(r.tcr.full)}</td>
                      <td className="n">{f2(r.tcr.blind)}</td>
                      <td className="n">
                        {signed(r.multimodal_gain.gain)} ({ci(r.multimodal_gain.ci95)})
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p>
            PathVQA&apos;s test and train images come from the same textbooks. When another panel of
            the same figure is in the corpus, its caption alone is enough. For the other two thirds
            of images, captions alone are wrong more often than right, and the image is what keeps
            the full system accurate.
          </p>
        </>
      )}

      <h2>How far to trust this</h2>
      <p>
        The claim checker (Gemini) agreed with a careful second labelling on 44 of 50 decisions
        (Cohen&apos;s κ = 0.82), but that second labelling was not done by a pathologist, and it was
        measured on caption claims, not generated ones. The same model writes and checks the
        descriptions, although the checker never sees the image. About 80% of generated claims
        can&apos;t be checked at all, because PathVQA&apos;s answers are short. Treat the diagnostic
        numbers as provisional.
      </p>
      <p className="muted small">
        Full reports: eval/phase3, eval/phase4 and eval/phase5 in the{" "}
        <a href="https://github.com/yashmyeole/pathscope">repository</a>.
      </p>
    </div>
  );
}
