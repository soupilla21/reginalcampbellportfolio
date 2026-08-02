import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const articles = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/articles' }),
  schema: z.object({
    title: z.string(),
    description: z.string().max(160, 'Keep meta descriptions close to ~155 characters.'),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    ogImage: z.string(),
    ogImageAlt: z.string(),
    tags: z.array(z.enum(['ai-governance', 'enterprise-transformation', 'delivery'])).optional(),
    draft: z.boolean().default(true),
    canonical: z.string().url().optional(),
  }),
});

export const collections = { articles };
