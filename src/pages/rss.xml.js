import rss from "@astrojs/rss";
import { getPublishedArticles } from "../lib/articles";

export async function GET(context) {
  const articles = (await getPublishedArticles()).sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf()
  );

  return rss({
    title: "Reginal Campbell | Articles",
    description: "Essays by Reginal Campbell on leadership, hiring and careers, AI and technology, and the personal side of work.",
    site: context.site,
    items: articles.map((article) => ({
      title: article.data.title,
      description: article.data.description,
      pubDate: article.data.pubDate,
      link: `/articles/${article.id}`,
    })),
  });
}
