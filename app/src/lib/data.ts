import { readFile } from "node:fs/promises";
import path from "node:path";

import gallery from "../../data/gallery.json";
import summary from "../../data/summary.json";

import type { QueryRecord } from "./shared";

export * from "./shared";

export type GalleryItem = { id: string; kind: string };

export const galleryItems = gallery as GalleryItem[];
export const results = summary;

const ID = /^[0-9a-f]{32}$/;

export async function loadQuery(id: string): Promise<QueryRecord | null> {
  if (!ID.test(id)) return null;
  try {
    const file = path.join(process.cwd(), "data", "queries", `${id}.json`);
    return JSON.parse(await readFile(file, "utf8")) as QueryRecord;
  } catch {
    return null;
  }
}
