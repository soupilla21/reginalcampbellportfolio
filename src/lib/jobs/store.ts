// Append-and-commit for the two files the website writes. Each write re-reads the file
// for its current SHA, applies the change, validates the whole file, and retries up to
// 3 times when someone else committed in between (spec §15).
import { ConfigError, readJsonWithSha, writeJson } from "./github";

export class InvalidFileError extends Error {}

export async function appendWithRetry<T>(opts: {
  path: string;
  empty: () => T;
  validate: (file: unknown) => string[];
  apply: (file: T) => T;
  message: string;
}): Promise<T> {
  for (let attempt = 0; attempt < 4; attempt++) {
    const current = await readJsonWithSha<T>(opts.path);
    if (current && opts.validate(current.data).length) {
      console.error(`${opts.path} failed validation; refusing to write`);
      throw new InvalidFileError(opts.path);
    }
    const next = opts.apply(current?.data ?? opts.empty());
    const errs = opts.validate(next);
    if (errs.length) {
      console.error(`${opts.path}: new content failed validation (${errs.length} error(s))`);
      throw new InvalidFileError(opts.path);
    }
    if ((await writeJson(opts.path, next, current?.sha ?? null, opts.message)) === "ok") return next;
    await new Promise((r) => setTimeout(r, 200 + attempt * 300));
  }
  throw new ConflictError();
}

export class ConflictError extends Error {}
export { ConfigError };
