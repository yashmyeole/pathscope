import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import QueryView from "@/components/QueryView";
import { galleryItems, loadQuery } from "@/lib/data";

export const dynamicParams = false;

export function generateStaticParams() {
  return galleryItems.map((x) => ({ id: x.id }));
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  return { title: `PathScope · ${id.slice(0, 8)}` };
}

export default async function QueryPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const rec = await loadQuery(id);
  if (!rec) notFound();

  const i = galleryItems.findIndex((x) => x.id === id);
  const prev = galleryItems[(i - 1 + galleryItems.length) % galleryItems.length].id;
  const next = galleryItems[(i + 1) % galleryItems.length].id;

  return (
    <>
      <p className="crumbs">
        <Link href="/">← All images</Link>
        <span className="muted"> · </span>
        <Link href={`/q/${prev}`}>Previous</Link>
        <span className="muted"> · </span>
        <Link href={`/q/${next}`}>Next</Link>
      </p>
      <QueryView rec={rec} />
    </>
  );
}
