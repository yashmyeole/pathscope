import type { Neighbour } from "./shared";

const corpus = process.env.QDRANT_COLLECTION || "pathscope_plip_v1";
const queries = process.env.QDRANT_QUERY_COLLECTION || `${corpus}_queries`;

type Point = {
  id: string;
  score: number;
  payload: { image_id: string; text: string | null; source: string; image_type: string };
};

// Point ids are the images' pixel MD5s, which Qdrant stores as UUIDs.
const uuid = (hex: string) =>
  `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;

export async function nearest(id: string, k = 5): Promise<{ neighbours: Neighbour[]; ms: number }> {
  const url = process.env.QDRANT_URL;
  if (!url) throw new Error("QDRANT_URL is not set");

  const res = await fetch(`${url.replace(/\/$/, "")}/collections/${corpus}/points/query`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "api-key": process.env.QDRANT_API_KEY ?? "",
    },
    body: JSON.stringify({
      query: uuid(id),
      lookup_from: { collection: queries },
      limit: k,
      with_payload: ["image_id", "text", "source", "image_type"],
    }),
    cache: "no-store",
    signal: AbortSignal.timeout(4000),
  });
  if (!res.ok) throw new Error(`Qdrant ${res.status}: ${(await res.text()).slice(0, 200)}`);

  const body = (await res.json()) as { result: { points: Point[] }; time: number };
  const neighbours = body.result.points.map((p, i) => ({
    rank: i + 1,
    id: p.payload.image_id,
    score: Math.round(p.score * 1000) / 1000,
    source: p.payload.source,
    kind: p.payload.image_type,
    caption: p.payload.text ?? "",
  }));
  return { neighbours, ms: Math.round(body.time * 1000) };
}
