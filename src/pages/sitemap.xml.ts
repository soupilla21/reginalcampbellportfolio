import type { APIRoute } from "astro";
import { getCollection } from "astro:content";

export const GET: APIRoute = async ({ site }) => {
  const base = site?.toString().replace(/\/$/, "") ?? "https://reginalcampbell.com";
  const articles = await getCollection("articles", ({ data }) => !data.draft);

  const urls = [
    { loc: `${base}/`, changefreq: "monthly", priority: "1.0" },
    { loc: `${base}/articles`, changefreq: "monthly", priority: "0.8" },
    ...articles.map((article) => ({
      loc: `${base}/articles/${article.id}`,
      changefreq: "monthly",
      priority: "0.7",
      lastmod: (article.data.updatedDate ?? article.data.pubDate).toISOString().split("T")[0],
    })),
  ];

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls
  .map(
    (u) => `  <url>
    <loc>${u.loc}</loc>${u.lastmod ? `\n    <lastmod>${u.lastmod}</lastmod>` : ""}
    <changefreq>${u.changefreq}</changefreq>
    <priority>${u.priority}</priority>
  </url>`
  )
  .join("\n")}
</urlset>
`;

  return new Response(body, {
    headers: { "Content-Type": "application/xml; charset=utf-8" },
  });
};
