import Link from "next/link";

import { galleryItems, imageUrl, results } from "@/lib/data";

const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);

const groups = [
  { kind: "micrograph", title: "Micrographs" },
  { kind: "photo", title: "Gross and clinical photos" },
  { kind: "diagram", title: "Diagrams and X-rays" },
];

export default function Home() {
  const p4 = results.phase4;
  const p5 = results.phase5;
  const noTwin = p5.byLabelMatch?.false;

  return (
    <>
      <h1>Do captions of similar pathology images describe a new one?</h1>
      <p className="lead">
        PathScope finds the five most similar images to a pathology image (PLIP embeddings in
        Qdrant) and asks whether their captions are true of it. Every caption is broken into
        single facts, and each fact is checked against what the PathVQA dataset records about the
        new image.
      </p>

      <div className="stats">
        <div className="stat">
          <div className="num">{pct(p4.overall.tcr)}</div>
          <p>
            of the facts in the 5 nearest captions that could be checked were true of the new image
            ({p4.queries} test images).
          </p>
        </div>
        <div className="stat">
          <div className="num">
            {pct(p5.conditions.full.tcr)} <span className="muted small">vs</span>{" "}
            {pct(p5.conditions.blind.tcr)}
          </div>
          <p>
            A description written from the captions <em>and</em> the image, against one written
            from the captions alone.
          </p>
        </div>
        {noTwin && (
          <div className="stat">
            <div className="num">{pct(noTwin.tcr.blind)}</div>
            <p>
              Captions alone, when the corpus has no copy of the same textbook figure (
              {noTwin.queries} of {p5.queries} images). With the image it stays at{" "}
              {pct(noTwin.tcr.full)}.
            </p>
          </div>
        )}
      </div>
      <p className="small">
        <Link href="/results">All results, and how far to trust them →</Link>
      </p>

      <h2>How it works</h2>
      <ol className="steps">
        <li>Every image in a 10,141-image corpus (PathVQA train, ARCH) is embedded with PLIP and stored in Qdrant.</li>
        <li>For a test image, Qdrant returns the five nearest corpus images and their captions.</li>
        <li>Gemini writes a short description from the image and those captions.</li>
        <li>The description is split into facts, and each one is checked against the image&apos;s own PathVQA answers.</li>
      </ol>

      <h2>Pick an image</h2>
      <p className="muted small">
        {galleryItems.length} test images. None of them, or a near-duplicate, is in the corpus.
      </p>
      {groups.map((g) => {
        const items = galleryItems.filter((x) => x.kind === g.kind);
        if (!items.length) return null;
        return (
          <section key={g.kind}>
            <h3 style={{ margin: "22px 0 8px" }}>
              {g.title} <span className="muted">({items.length})</span>
            </h3>
            <div className="grid">
              {items.map((x) => (
                <Link key={x.id} href={`/q/${x.id}`} className="tile" aria-label={`Open ${g.title} image`}>
                  <img src={imageUrl(x.id)} alt="" loading="lazy" decoding="async" />
                </Link>
              ))}
            </div>
          </section>
        );
      })}
    </>
  );
}
