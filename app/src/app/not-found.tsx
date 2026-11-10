import Link from "next/link";

export default function NotFound() {
  return (
    <>
      <h1>Not found</h1>
      <p>
        That image isn&apos;t in the demo. <Link href="/">Back to all images</Link>.
      </p>
    </>
  );
}
