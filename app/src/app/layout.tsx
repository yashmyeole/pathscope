import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "PathScope",
  description:
    "Do the captions of similar-looking pathology images describe a new one? A retrieval study on PathVQA and ARCH.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="banner">Research project. Not for clinical use.</div>
        <header className="top">
          <Link href="/" className="brand">
            PathScope
          </Link>
          <nav>
            <Link href="/">Images</Link>
            <Link href="/results">Results</Link>
            <a href="https://github.com/yashmyeole/pathscope">Code</a>
          </nav>
        </header>
        <main>{children}</main>
        <footer>
          CS594 Vector Databases, University of Illinois Chicago, Fall 2026. Images from PathVQA
          (MIT) and ARCH (CC BY-NC-SA 4.0), shown for non-commercial research.
        </footer>
      </body>
    </html>
  );
}
