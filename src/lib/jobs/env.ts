// Server-only settings for /jobs. Values come from Vercel environment variables
// (Production): GITHUB_DATA_TOKEN, DATA_REPO, JOBS_PASSWORD_REGGIE, JOBS_PASSWORD_YAHSH. Never import this
// file from client code.

function read(name: string): string | undefined {
  const fromProcess = typeof process !== "undefined" ? process.env?.[name] : undefined;
  const fromVite = ((import.meta as any).env ?? {})[name] as string | undefined;
  const v = (fromProcess ?? fromVite ?? "").trim();
  return v === "" ? undefined : v;
}

export const jobsEnv = {
  // One password per reviewer: JOBS_PASSWORD_REGGIE and JOBS_PASSWORD_YAHSH.
  // A reviewer whose variable is missing can't sign in. Two identical passwords are
  // refused, since either person could then sign in as the other.
  passwordFor: (reviewer: string) => {
    const own = read(`JOBS_PASSWORD_${reviewer.toUpperCase()}`);
    const others = ["Reggie", "Yahsh"].filter((r) => r !== reviewer).map((r) => read(`JOBS_PASSWORD_${r.toUpperCase()}`));
    if (!own || others.includes(own)) return undefined;
    return own;
  },
  dataToken: () => read("GITHUB_DATA_TOKEN"),
  dataRepo: () => {
    const repo = read("DATA_REPO");
    return repo && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo) ? repo : undefined;
  },
  dataBranch: () => {
    const b = read("DATA_BRANCH") ?? "main";
    return /^[A-Za-z0-9._\/-]+$/.test(b) ? b : "main";
  },
  // Local testing only: point the GitHub client at a mock server. Ignored in production builds.
  apiBase: () => (((import.meta as any).env?.DEV === true) && read("GITHUB_API_URL")) || "https://api.github.com",
};

export const REVIEWERS = ["Reggie", "Yahsh"] as const;
export type Reviewer = (typeof REVIEWERS)[number];
export const isReviewer = (v: unknown): v is Reviewer => typeof v === "string" && (REVIEWERS as readonly string[]).includes(v);
