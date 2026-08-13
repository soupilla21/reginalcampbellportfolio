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

const artifacts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/artifacts' }),
  schema: z
    .object({
      title: z.string(),
      description: z.string().max(160),
      ogImage: z.string(),
      ogImageAlt: z.string(),
      draft: z.boolean().default(true),
      lastReviewed: z.coerce.date(),
      version: z.string().default('1.0'),
      artifactType: z.enum(['risk-assessment', 'policy', 'checklist']).optional(),
      frameworkAlignment: z
        .array(z.enum(['nist-ai-rmf', 'iso-42001', 'eu-ai-act']))
        .optional(),
      anchorScenario: z.enum(['leadership-agent', 'helios']).optional(),
      order: z.number().int().optional(),
      downloadPath: z.string().optional(),
      downloadFormat: z.enum(['pdf', 'docx', 'xlsx']).optional(),
      canonical: z.string().url().optional(),
    })
    .refine(
      (d) => (d.downloadPath === undefined) === (d.downloadFormat === undefined),
      { message: 'downloadPath and downloadFormat must be set together' }
    )
    .refine((d) => d.lastReviewed <= new Date(), {
      message: 'lastReviewed cannot be in the future',
    }),
});

export const collections = { articles, artifacts };
