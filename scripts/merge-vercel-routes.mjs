// Runs after `astro build`. With the @astrojs/vercel adapter the build writes Vercel's
// Build Output (.vercel/output/config.json), and Vercel's docs don't promise that the
// redirects, rewrites and headers in vercel.json are still applied on top of it.
// This folds them into config.json with Vercel's own converter, so /ask/, the client
// portal rewrites and the first-90-days noindex header keep working either way.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { getTransformedRoutes } from "@vercel/routing-utils";

const CONFIG = ".vercel/output/config.json";
if (!existsSync(CONFIG)) {
  console.log("merge-vercel-routes: no .vercel/output/config.json (adapter not active); nothing to do.");
  process.exit(0);
}
const v = JSON.parse(readFileSync("vercel.json", "utf8"));
const { routes, error } = getTransformedRoutes({
  redirects: v.redirects, rewrites: v.rewrites, headers: v.headers,
  cleanUrls: v.cleanUrls, trailingSlash: v.trailingSlash,
});
if (error) {
  console.error("merge-vercel-routes: vercel.json could not be converted:", error.message ?? error);
  process.exit(1);
}
const config = JSON.parse(readFileSync(CONFIG, "utf8"));
const split = routes.findIndex((r) => r.handle === "filesystem");
const before = split === -1 ? routes : routes.slice(0, split); // redirects + headers
const after = split === -1 ? [] : routes.slice(split + 1);     // rewrites (after static files)
const key = (r) => JSON.stringify(r);
const existing = new Set(config.routes.map(key));
const addBefore = before.filter((r) => !existing.has(key(r)));
const addAfter = after.filter((r) => !existing.has(key(r)));
config.routes = [...addBefore, ...config.routes, ...addAfter];
writeFileSync(CONFIG, JSON.stringify(config, null, 2));
console.log(`merge-vercel-routes: added ${addBefore.length} redirect/header and ${addAfter.length} rewrite routes from vercel.json.`);
