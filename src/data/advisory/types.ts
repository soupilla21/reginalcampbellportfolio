// Shared types for advisory briefings (/advisory/<slug>).
//
// A briefing is two things kept apart on purpose:
//   1. a document manifest: the approved source files, their metadata and
//      publication status. Adding or replacing a file is a manifest edit.
//   2. narrative content: the chapter copy and diagram data, each item
//      carrying the source document(s) it comes from.
// Layout code reads both and never hard-codes either.

// Evidence tags exactly as the source package defines them:
// [S] simulated finding, [E] estimate, [A] assumption, [J] judgment.
// "R" marks a recommendation stated in the package.
export type EvidenceTag = "S" | "E" | "A" | "J" | "R";

// A pointer into the package: document ID plus optional section label.
export interface SourceRef {
  doc: string; // manifest id, e.g. "08"
  section?: string;
}

// Publication status of a single file.
//   approved   may be served in any build where the briefing itself is built
//   private    never copied into this repo; listed only so the omission is documented
export type FileStatus = "approved" | "private";

export interface ManifestDocument {
  id: string; // short id used in source refs, e.g. "08"
  code: string; // the package's own document code, e.g. "HRHS-AI-08"
  title: string;
  description: string; // one or two sentences, executive register
  purpose: string; // "Use it to ..." from the Reader's Guide
  version: string;
  date: string; // as printed on the document
  pages: number;
  group: "start" | "reasoning" | "execution" | "framework";
  order: number; // recommended reading order (Reader's Guide, v2.1)
  file: string; // file name under advisory-packages/<slug>/<version>/files
  preview: string; // file name under .../previews (first page, rendered from the PDF)
  previewSize: { width: number; height: number };
  original?: { file: string; label: string }; // e.g. the .pptx behind the deck PDF
  status: FileStatus;
  featured?: boolean; // shown in the persistent "direct access" bar
}

// Briefing-level publication state.
//   review   builds locally and on Vercel preview deployments only; never on
//            the production deployment; always noindex; not in the sitemap.
//   public   builds everywhere, indexable, listed in the sitemap.
export type BriefingStatus = "review" | "public";

export interface BriefingMeta {
  slug: string;
  packageVersion: string;
  status: BriefingStatus;
  title: string; // <title>
  description: string; // meta description
  socialImage: string; // file name under .../previews
}
