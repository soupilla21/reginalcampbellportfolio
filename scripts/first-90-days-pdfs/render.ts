// Renders the two leave-behind documents for /advisory/first-90-days as HTML,
// from the same content module the page uses, so the figures always match.
//   npx tsx scripts/first-90-days-pdfs/render.ts <outDir>
// then: python3 scripts/first-90-days-pdfs/print.py <outDir>
// which writes the PDFs into advisory-packages/first-90-days/files/.
import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import * as d from "../../src/data/playbooks/first-90-days";

const out = process.argv[2] ?? "/tmp/first-90-days-pdf";
mkdirSync(out, { recursive: true });

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const rag = { g: "Green", a: "Yellow", r: "Red" } as const;
const dot = (r: "g" | "a" | "r") => `<span class="dot ${r}"></span>${rag[r]}`;
const cap = d.steering.capacity;

const css = `
@page { size: Letter; margin: 0.55in 0.6in; }
@page wide { size: Letter landscape; margin: 0.5in 0.55in; }
* { box-sizing: border-box; }
body { font-family: Inter, "Helvetica Neue", Arial, sans-serif; color: #1d2733; font-size: 9.6pt; line-height: 1.45; margin: 0; }
h1 { font-size: 24pt; letter-spacing: -0.02em; margin: 0; color: #0f1a28; line-height: 1.1; }
h2 { font-size: 13pt; margin: 0 0 6pt; color: #0f1a28; }
h3 { font-size: 8pt; text-transform: uppercase; letter-spacing: 0.12em; color: #5b6573; margin: 0 0 4pt; font-weight: 600; }
p { margin: 0 0 5pt; }
ul, ol { margin: 0; padding-left: 13pt; } li { margin-bottom: 2.5pt; }
.eyebrow { font-size: 7.5pt; letter-spacing: 0.16em; text-transform: uppercase; color: #3d6b91; margin-bottom: 6pt; }
.sub { color: #3d6b91; font-size: 12pt; margin: 4pt 0 10pt; }
.hook { border-left: 3px solid #c08a3e; padding-left: 9pt; font-size: 11pt; color: #0f1a28; margin: 0 0 12pt; }
.tag { font-size: 6.5pt; letter-spacing: 0.1em; text-transform: uppercase; border: 1px solid #b9c2cc; border-radius: 99px; padding: 1pt 6pt; color: #5b6573; vertical-align: middle; margin-left: 6pt; font-weight: 500; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10pt; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12pt; }
.box { border: 1px solid #d4dae1; border-radius: 6pt; padding: 8pt 9pt; }
.warm { background: #fff6e8; border-color: #e3c08d; }
.phase .days { font-size: 7.5pt; color: #8a5d22; letter-spacing: 0.1em; text-transform: uppercase; }
.phase .t { font-weight: 700; font-size: 10.5pt; color: #0f1a28; margin: 1pt 0 3pt; }
.phase .o { color: #8a5d22; font-weight: 600; margin-bottom: 4pt; }
.sec { margin-top: 12pt; }
table { width: 100%; border-collapse: collapse; font-size: 8.4pt; }
th, td { text-align: left; vertical-align: top; padding: 4pt 5pt; border-bottom: 1px solid #e1e6eb; }
thead th { font-size: 7.4pt; color: #5b6573; border-bottom: 1.5px solid #b9c2cc; font-weight: 600; }
tbody th { font-weight: 600; color: #0f1a28; }
tr.hl > * { background: #fff1db; }
.raci td { text-align: center; font-family: "JetBrains Mono", Menlo, monospace; color: #5b6573; }
.raci td.a { color: #8a5d22; font-weight: 700; }
.dot { display: inline-block; width: 7pt; height: 7pt; border-radius: 50%; margin-right: 3pt; vertical-align: -0.5pt; }
.dot.g { background: #3f9b63; } .dot.a { background: #e3b51c; } .dot.r { background: #c24d40; }
.foot { margin-top: 12pt; padding-top: 6pt; border-top: 1px solid #d4dae1; font-size: 7.4pt; color: #5b6573; display: flex; justify-content: space-between; gap: 12pt; }
.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }
.wide { page: wide; }
.tight > *:not(.foot) { zoom: 0.88; }
body.one { zoom: 0.84; }
.callout { border-left: 3px solid #c08a3e; background: #fff6e8; padding: 5pt 8pt; margin-top: 8pt; font-size: 8.6pt; }
.ask { background: #fff6e8; border: 1px solid #e3c08d; border-radius: 6pt; padding: 6pt 9pt; margin: 6pt 0 9pt; font-weight: 600; color: #0f1a28; }
.ask span { display: block; font-size: 6.8pt; letter-spacing: 0.14em; text-transform: uppercase; color: #8a5d22; font-weight: 600; }
.opt { border: 1px solid #d4dae1; border-radius: 6pt; padding: 7pt; font-size: 8.3pt; }
.opt.rec { border: 1.5pt solid #c08a3e; background: #fffaf1; }
.opt b.n { display: block; font-size: 7pt; letter-spacing: 0.1em; text-transform: uppercase; color: #5b6573; }
.opt dl { margin: 4pt 0 0; } .opt dt { color: #5b6573; font-size: 7.4pt; margin-top: 3pt; } .opt dd { margin: 0; }
.kit-head { display: flex; align-items: baseline; justify-content: space-between; border-bottom: 2px solid #0f1a28; padding-bottom: 5pt; margin-bottom: 9pt; }
.kit-head .n { font-size: 8pt; letter-spacing: 0.14em; text-transform: uppercase; color: #8a5d22; }
.meta { font-size: 8pt; color: #5b6573; margin-bottom: 8pt; }
.bar { display: flex; height: 13pt; border-radius: 3pt; overflow: hidden; background: #eef1f4; font-size: 7pt; font-weight: 700; }
.bar span { display: flex; align-items: center; padding: 0 4pt; white-space: nowrap; }
`;

const footer = (label: string) =>
  `<div class="foot"><span>${esc(label)} · Reginal Campbell, PMP · reginalcampbell.com/advisory/${d.meta.slug}</span><span>Illustrative examples. ${esc(d.meta.date)}</span></div>`;

// ── One-page summary ──
const evid = d.evidence;
const summary = `<!doctype html><html><head><meta charset="utf-8"><title>90 Days to a Decision-Ready Technology Portfolio: One-Page Summary</title><style>${css}</style></head><body class="one">
<div class="eyebrow">90-day plan · Reginal Campbell, PMP · Illustrative scenario</div>
<h1>${esc(d.hero.title)}</h1>
<p class="sub">${esc(d.hero.subtitle)}</p>
<div class="grid3">
${d.phases.map((p, i) => { const o = d.hero.outcomes[i]; return `<div class="box phase"><div class="days">Day ${o.day} · ${p.days}</div><div class="t" style="font-size:13pt">${o.word}</div><div class="o">${esc(o.text)}</div><p style="margin:0"><b>${esc(p.title)}.</b> ${esc(p.output)}</p></div>`; }).join("")}
</div>
<div class="sec grid2">
  <div class="box warm">
    <h3>The Day 60 decision</h3>
    <p>Programs B (spend analytics release) and C (supplier and contract data platform) both need the 6-person integration team in Weeks 10–15: ${cap.b.toLocaleString()} + ${cap.c} hours against ${cap.available.toLocaleString()} available, a ${cap.b + cap.c - cap.available}-hour shortfall.</p>
    <p><b>Options:</b> sequence (C slips, audit finding likely), phase scope (B launches core scope, C does audit-critical work first), or add capacity (about $72K, over the 10% threshold).</p>
    <p style="margin:0"><b>Recommended:</b> phase scope. Both external commitments hold at no added cost, with a ${cap.available - cap.phasedB - cap.phasedC}-hour buffer and a contractor fallback to decide by Day 74.</p>
  </div>
  <div class="box">
    <h3>Technical delivery path</h3>
    <ol>${d.deliveryPath.map((n) => `<li><b>${esc(n.name)}.</b> ${esc(n.done)}</li>`).join("")}</ol>
  </div>
</div>
<div class="sec grid2">
  <div><h3>What each seat gets</h3>
    <table><tbody>${d.roles.map((r) => `<tr><th>${esc(r.name)}</th><td>${esc(r.get)}</td></tr>`).join("")}</tbody></table>
  </div>
  <div><h3>What you can hold me to</h3>
    <ul>${d.agreement.commitments.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>
  </div>
</div>
<div class="sec box"><h3>Evidence from my work</h3><p style="margin:0"><b>${esc(evid.title)}</b> (${esc(evid.org)}), ${esc(evid.role)}. ${evid.results.map((r) => `${r.value} ${esc(r.label)}`).join("; ")}.</p></div>
<p class="hook" style="margin-top:10pt;font-size:10.5pt">${esc(d.agreement.close)} ${esc(d.agreement.invitation)}</p>
<p style="font-size:7.4pt;color:#5b6573">${esc(d.disclosure)}</p>
${footer("One-page summary")}
</body></html>`;

// ── Starter kit ──
const page = (n: number, name: string, meta: string, body: string, wide = false, tight = false) =>
  `<section class="page${wide ? " wide" : ""}${tight ? " tight" : ""}"><div class="kit-head"><h2 style="margin:0">${esc(name)}<span class="tag">Illustrative</span></h2><span class="n">Artifact ${n}</span></div><p class="meta">${esc(meta)}</p>${body}${footer("Governance Starter Kit")}</section>`;
const a = (id: string) => d.artifacts.find((x) => x.id === id)!;
const metaOf = (id: string) => { const x = a(id); return `For: ${x.who}. When: ${x.when}. Supports: ${x.supports}.`; };

const kitPages = [
  `<section class="page"><div class="eyebrow">Governance Starter Kit</div><h1>Six artifacts, filled in</h1><p class="sub">From "90 Days to a Decision-Ready Technology Portfolio" · Reginal Campbell, PMP</p>
   <p class="hook">${esc(d.hero.subtitle)}</p>
   <p>Every artifact in this kit supports a decision, and every one uses the same sample portfolio and numbers:</p>
   <table><thead><tr><th>Program</th><th>Sponsor</th><th>Budget</th><th>Why it matters</th></tr></thead><tbody>${d.programs.map((p) => `<tr><th>${p.key}. ${esc(p.name)}</th><td>${esc(p.sponsor)}</td><td>${p.budget}</td><td>${esc(p.note)}</td></tr>`).join("")}</tbody></table>
   <p style="margin-top:8pt">Illustrative scenario set in healthcare supply chain and spend technology. One 6-person integration engineering team serves B and C. On Day 47 the shared RAID log shows both booking it for Weeks 10–15. The artifacts follow that conflict from first signal to executive decision.</p>
   <div class="sec"><h3>Contents</h3><ol>${["Decision rights matrix", "Stoplight status card and escalation thresholds", "RAID log excerpt", "Steering pre-read", "Executive one-pager", "Decision log"].map((t) => `<li>${t}</li>`).join("")}</ol></div>
   <p style="margin-top:12pt;font-size:8pt;color:#5b6573">${esc(d.disclosure)}</p>${footer("Governance Starter Kit")}</section>`,

  page(1, "Decision rights matrix", metaOf("art-raci"),
    `<p class="meta">R does the work · A decides (one per row) · C consulted · I informed</p>
     <table class="raci"><thead><tr><th>Decision</th>${d.raciRoles.map((r) => `<th style="text-align:center">${esc(r)}</th>`).join("")}</tr></thead><tbody>
     ${d.raci.map((r) => `<tr class="${r.highlight ? "hl" : ""}"><th>${esc(r.decision)}</th>${r.cells.map((c) => `<td class="${c.includes("A") ? "a" : ""}">${c}</td>`).join("")}</tr>`).join("")}</tbody></table>
     <div class="callout">${esc(d.raciNote)}</div>`, true),

  page(2, "Stoplight status card: no watermelon statuses", metaOf("art-rag"),
    `<p>A watermelon status is green on the outside and red on the inside. These criteria make status a measurement, not a mood. ${esc(d.tierNote)}</p>
     <table><thead><tr><th></th><th>${dot("g")}</th><th>${dot("a")}</th><th>${dot("r")}</th></tr></thead><tbody>
     ${d.ragRows.map((r) => `<tr><th>${r.dim}</th><td>${esc(r.g)}</td><td>${esc(r.a)}</td><td>${esc(r.r)}</td></tr>`).join("")}
     <tr class="hl"><th>What it means</th><td>${esc(d.ragMeaning.g)}</td><td>${esc(d.ragMeaning.a)}</td><td>${esc(d.ragMeaning.r)}</td></tr></tbody></table>
     <div class="sec grid2"><div><h3>Four rules</h3><ol>${d.ragRules.map((r) => `<li>${esc(r)}</li>`).join("")}</ol></div>
     <div><h3>Escalation thresholds</h3><table><tbody>${d.thresholds.map((t) => `<tr><th>${esc(t.trigger)}${t.limit !== "Any" ? ` (${t.limit.toLowerCase()})` : ""}</th><td>${esc(t.to)}</td><td>${esc(t.within)}</td></tr>`).join("")}</tbody></table></div></div>`, true),

  page(3, "RAID log excerpt, as of Day 52", metaOf("art-raid"),
    `<table><thead><tr><th>ID</th><th>Type</th><th>Program</th><th>Description</th><th>Owner</th><th>Impact</th><th>Status</th><th>Next action</th></tr></thead><tbody>
     ${d.raid.map((r) => `<tr class="${r.key ? "hl" : ""}"><th>${r.id}</th><td>${r.type}</td><td>${esc(r.prog)}</td><td>${esc(r.text)}</td><td>${esc(r.owner)}</td><td>${esc(r.impact)}</td><td style="white-space:nowrap">${dot(r.rag)}</td><td>${esc(r.next)}</td></tr>`).join("")}</tbody></table>
     <div class="callout">${esc(d.raidNote)}</div>`, true),

  page(4, "Steering pre-read · Steering 2, Day 60", metaOf("art-preread"),
    `<div class="ask"><span>Decision requested</span>${esc(d.steering.decision)}</div>
     <div class="grid2"><div><h3>Situation</h3><ul>${d.steering.situation.map((s) => `<li>${esc(s)}</li>`).join("")}</ul><p style="margin-top:4pt;color:#5b6573">${esc(d.steering.cause)}</p></div>
     <div><h3>Integration engineering, Weeks 10–15 (hours)</h3>
       <p style="margin:4pt 0 2pt;font-size:7.4pt">Committed</p><div class="bar"><span style="width:${(cap.b / (cap.b + cap.c)) * 100}%;background:#3d6b91;color:#fff">B ${cap.b}</span><span style="width:${(cap.c / (cap.b + cap.c)) * 100}%;background:#7fabcf">C ${cap.c}</span></div>
       <p style="margin:4pt 0 2pt;font-size:7.4pt">Available</p><div class="bar"><span style="width:${(cap.available / (cap.b + cap.c)) * 100}%;background:#c9d3dd">${cap.available.toLocaleString()}</span><span style="flex:1;background:#c24d40;color:#fff">−${cap.b + cap.c - cap.available}</span></div>
       <p style="margin:4pt 0 2pt;font-size:7.4pt">Option 2</p><div class="bar"><span style="width:${(cap.phasedB / (cap.b + cap.c)) * 100}%;background:#3d6b91;color:#fff">B ${cap.phasedB}</span><span style="width:${(cap.phasedC / (cap.b + cap.c)) * 100}%;background:#7fabcf">C ${cap.phasedC}</span><span style="width:${((cap.available - cap.phasedB - cap.phasedC) / (cap.b + cap.c)) * 100}%;background:#e5d2b2">+${cap.available - cap.phasedB - cap.phasedC}</span></div>
     </div></div>
     <div class="sec"><h3>Options</h3><div class="grid3">${d.steering.options.map((o) => `<div class="opt${o.recommended ? " rec" : ""}"><b class="n">Option ${o.n}${o.recommended ? " · Recommended" : ""}</b><b style="font-size:10pt">${o.name}</b><p style="margin:2pt 0 0">${esc(o.short)}</p><dl><dt>Hours</dt><dd>${esc(o.hours)}</dd><dt>Schedule</dt><dd>${esc(o.schedule)}</dd><dt>Cost</dt><dd>${esc(o.cost)}</dd><dt>Risk</dt><dd>${esc(o.risk)}</dd><dt>Reversible?</dt><dd>${esc(o.reversible)}</dd></dl></div>`).join("")}</div></div>
     <div class="sec grid2"><div><h3>Recommendation</h3><p>${esc(d.steering.recommendation)}</p></div>
     <div><h3>What we need from you</h3><ul>${d.steering.asks.map((x) => `<li><b>${esc(x.who)}:</b> ${esc(x.ask)}</li>`).join("")}</ul><p style="margin-top:4pt"><b>If we don't decide on Day 60:</b> ${esc(d.steering.noDecision)}</p></div></div>`, false, true),

  page(5, "Executive one-pager · Day 90", metaOf("art-exec"),
    `<div class="ask"><span>Decision needed</span>${esc(d.exec.decision)}</div>
     <table><thead><tr><th>Program</th><th>Status</th><th>Confidence</th><th>Trend</th><th>One line</th></tr></thead><tbody>${d.exec.programs.map((x) => { const p = d.programs.find((q) => q.key === x.key)!; return `<tr><th>${x.key}. ${esc(p.name)}</th><td style="white-space:nowrap">${dot(x.rag)}</td><td>${x.confidence}</td><td>${x.trend}</td><td>${esc(x.line)}</td></tr>`; }).join("")}</tbody></table>
     <div class="sec grid3"><div><h3>Value at risk</h3><p>${esc(d.exec.valueAtRisk)}</p></div><div><h3>Top three risks</h3><ol>${d.exec.risks.map((r) => `<li>${esc(r.text)} <span style="color:#5b6573">Owner: ${esc(r.owner)}.</span></li>`).join("")}</ol></div><div><h3>What changed since Steering 2</h3><p>${esc(d.exec.changed)}</p></div></div>
     <div class="sec grid3">${d.exec.scorecard.map((s) => `<div class="box"><h3>${s.area}</h3><div style="font-size:16pt;font-weight:700;color:#0f1a28">${s.value}</div><div style="font-size:8pt;color:#5b6573">${esc(s.text)}</div></div>`).join("")}</div>
     <div class="sec"><h3>The four calls</h3><p><b>Proceed:</b> criteria met and controls in place. <b>Pivot:</b> the evidence favors a different approach. <b>Slow:</b> capacity or adoption is short, so scope is sequenced. <b>Stop:</b> criteria still unmet after one revision.</p></div>`),

  page(6, "Decision log, Days 1–90", metaOf("art-log"),
    `<table><thead><tr><th>#</th><th>Day</th><th>Forum</th><th>Decision</th><th>Decided by</th><th>Basis</th><th>Revisit if</th></tr></thead><tbody>
     ${d.decisions.map((x) => `<tr class="${x.key ? "hl" : ""}"><th>${x.n}</th><td>${x.day}</td><td>${esc(x.forum)}</td><td>${esc(x.decision)}</td><td>${esc(x.by)}</td><td>${esc(x.basis)}</td><td>${esc(x.revisit)}</td></tr>`).join("")}</tbody></table>
     <div class="callout">The "Revisit if" column is the point: a decision should change when the evidence changes, and everyone should know in advance what evidence that is.</div>
     <p class="hook" style="margin-top:14pt">${esc(d.agreement.close)}</p>`, true),
];

const kit = `<!doctype html><html><head><meta charset="utf-8"><title>Governance Starter Kit</title><style>${css}</style></head><body>${kitPages.join("")}</body></html>`;

writeFileSync(join(out, "summary.html"), summary);
writeFileSync(join(out, "kit.html"), kit);
console.log("wrote", out);
