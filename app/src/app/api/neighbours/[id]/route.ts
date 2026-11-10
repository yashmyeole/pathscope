import { nearest } from "@/lib/qdrant";

export const dynamic = "force-dynamic";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^[0-9a-f]{32}$/.test(id)) {
    return Response.json({ error: "expected a 32-character image id" }, { status: 400 });
  }
  const started = Date.now();
  try {
    const { neighbours, ms } = await nearest(id);
    return Response.json({ neighbours, qdrantMs: ms, totalMs: Date.now() - started });
  } catch (e) {
    console.error("qdrant lookup failed", id, e);
    return Response.json({ error: "vector search unavailable" }, { status: 503 });
  }
}
