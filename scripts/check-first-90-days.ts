// Consistency check for /advisory/first-90-days: the sample numbers must agree
// across every artifact.   npx tsx scripts/check-first-90-days.ts
import * as d from "../src/data/playbooks/first-90-days";
const c = d.steering.capacity;
const fails: string[] = [];
const ok = (cond: boolean, msg: string) => { if (!cond) fails.push(msg); };
ok(c.available === c.team * 40 * c.weeks * c.focus && c.team === 8 && c.focus === 0.75, "1,440 = 8 engineers x 40 h x 6 weeks x 75% focus");
ok(c.b - c.phasedB === 300 && c.c - c.phasedC === 120, "deferred work: 300 hours from B, 120 from C (420 total)");
ok(c.contractBilled === 2 * 40 * c.weeks && c.contract === c.contractBilled * c.focus, "contractors: 480 billed, 360 productive");
ok(c.b + c.c - c.available === 240, "shortfall is 240");
ok(c.available - c.phasedB - c.phasedC === 180, "option 2 buffer is 180");
ok(Math.round((72 / 600) * 100) === 12, "$72K is 12% of C's $600K");
ok(d.programs.find((p) => p.key === "C")!.budget === "$600K", "C budget $600K");
const text = JSON.stringify(d);
for (const s of ["1,440", "240", "1,260", "180", "Week 16", "Week 18", "Week 20", "Day 74", "$72K", "75%", "300 hours", "120 hours", "480 billed"]) ok(text.includes(s), `mentions ${s}`);
ok(d.raci.every((r) => r.cells.filter((x) => x.includes("A")).length === 1), "one A per RACI row");
ok(d.decisions.map((x) => x.n).join() === "001,002,003,004,005,006,007", "decision numbers in order");
ok(d.decisions.find((x) => x.n === "005")!.day === 60 && d.decisions.find((x) => x.n === "007")!.day === 90, "#005 at Day 60, #007 at Day 90");
ok(d.artifacts.length === 12, "12 artifacts");
const o = Object.fromEntries(d.steering.options.map((x) => [x.key, x.bars]));
ok(o.phase.capacity - o.phase.b - o.phase.c === 180, "option 2 bars leave 180");
ok(o.capacity.capacity - c.available === c.contract && o.capacity.capacity - o.capacity.b - o.capacity.c === 120, "contractor option: +360 productive hours, 120-hour buffer");
ok(!/Nothing goes red without going yellow/.test(text), "no absolute yellow-before-red rule (control failures go red immediately)");
ok(d.weeklyStatus.dims.find((x) => x.dim === "Scope")!.rag === "g", "approved scope change is green, matching the stoplight card");
ok(!/—/.test(text), "no em dashes in copy");
// Mixed audience: "RAG" also means retrieval-augmented generation to data teams.
ok(!/\bRAG\b|[Aa]mber/.test(text), 'status colors are green / yellow / red, never "RAG" or "amber"');
if (fails.length) { console.error("FAIL\n- " + fails.join("\n- ")); process.exit(1); }
console.log("All consistency checks pass.");
