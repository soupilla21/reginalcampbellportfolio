// Serves the leave-behind PDFs for /advisory/first-90-days. Same gate as the
// page: nothing is emitted on a production build while status is "review".
// Files live outside public/ in advisory-packages/first-90-days/files/.
import type { APIRoute, GetStaticPaths } from "astro";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { isBuiltStatus } from "../../../../data/playbooks";
import { meta, files } from "../../../../data/playbooks/first-90-days";

export const getStaticPaths: GetStaticPaths = () =>
  isBuiltStatus(meta.status) ? Object.values(files).map((file) => ({ params: { file } })) : [];

export const GET: APIRoute = ({ params }) => {
  const body = readFileSync(join(process.cwd(), "advisory-packages", meta.slug, "files", params.file!));
  return new Response(body, { headers: { "Content-Type": "application/pdf" } });
};
