export type Tier = "A" | "B" | "C";
export type Label = "supported" | "contradicted" | "silent";
export type Condition = "full" | "blind" | "random";

export type Neighbour = {
  rank: number;
  id: string;
  score: number;
  source: string;
  kind: string;
  caption: string;
};

export type Claim = {
  text: string;
  tier: Tier;
  label: Label;
  from: number | null;
  copied: boolean;
};

export type Generated = {
  description: string;
  captions: string[] | null;
  claims: Claim[];
};

export type QueryRecord = {
  id: string;
  kind: string;
  labelMatch: boolean;
  facts: { q: string; a: string }[];
  neighbours: Neighbour[];
  conditions: Partial<Record<Condition, Generated>>;
};

export const imageUrl = (id: string) => `/img/${id}.webp`;

export const sourceName: Record<string, string> = {
  pathvqa_train: "PathVQA",
  arch_books: "ARCH textbook",
  arch_pubmed: "ARCH PubMed",
};
