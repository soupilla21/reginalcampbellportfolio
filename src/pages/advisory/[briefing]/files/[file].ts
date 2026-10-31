// Serves an advisory briefing's approved files: source PDFs, the original
// PowerPoint, first-page previews and the social card.
//
// Files live outside public/ (advisory-packages/<slug>/<version>/). They are
// emitted into the build only for briefings that pass the publication gate in
// src/data/advisory/index.ts, and only when the manifest marks them approved.
// Bytes are copied unmodified: the documents are never rewritten.
import type { APIRoute, GetStaticPaths } from "astro";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { builtBriefings, packageDir } from "../../../../data/advisory";

const TYPES: Record<string, string> = {
  pdf: "application/pdf",
  pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  webp: "image/webp",
  png: "image/png",
};

export const getStaticPaths: GetStaticPaths = () => {
  const paths: { params: { briefing: string; file: string }; props: { path: string } }[] = [];
  for (const { meta, documents } of builtBriefings()) {
    const dir = packageDir(meta);
    const add = (name: string, sub: "files" | "previews") =>
      paths.push({ params: { briefing: meta.slug, file: name }, props: { path: join(dir, sub, name) } });
    for (const d of documents.filter((d) => d.status === "approved")) {
      add(d.file, "files");
      add(d.preview, "previews");
      if (d.original) add(d.original.file, "files");
    }
    add(meta.socialImage, "previews");
  }
  return paths;
};

export const GET: APIRoute = ({ props }) => {
  const path = (props as { path: string }).path;
  const ext = path.split(".").pop()!.toLowerCase();
  const body = readFileSync(join(process.cwd(), path));
  return new Response(body, { headers: { "Content-Type": TYPES[ext] ?? "application/octet-stream" } });
};
