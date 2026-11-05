// Serves the leave-behind PDFs and their first-page previews for /advisory/first-90-days. Same gate as the
// page: nothing is emitted on a production build while status is "review".
// Files live outside public/ in advisory-packages/first-90-days/files/.
import type { APIRoute, GetStaticPaths } from "astro";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { isBuiltStatus } from "../../../../data/playbooks";
import { meta, documents } from "../../../../data/playbooks/first-90-days";

const dir = (sub: string, name: string) => join("advisory-packages", meta.slug, sub, name);

export const getStaticPaths: GetStaticPaths = () =>
  isBuiltStatus(meta.status)
    ? documents.flatMap((d) => [
        { params: { file: d.file }, props: { path: dir("files", d.file) } },
        { params: { file: d.preview }, props: { path: dir("previews", d.preview) } },
      ])
    : [];

export const GET: APIRoute = ({ props }) => {
  const path = (props as { path: string }).path;
  const body = readFileSync(join(process.cwd(), path));
  const type = path.endsWith(".webp") ? "image/webp" : "application/pdf";
  return new Response(body, { headers: { "Content-Type": type } });
};
