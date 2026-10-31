// Behavior for /advisory/<slug> briefings. Progressive enhancement only:
// every chapter, panel and document is in the HTML and readable without this
// script. It adds tabs, scenario switching, the current-chapter marker, a
// full-screen PDF viewer on large screens and restrained reveal effects.
// Analytics use the site's existing data-ga-event click tracker; nothing
// here sends events directly except the viewer (which reuses gtag if present).

const root = document.querySelector<HTMLElement>("[data-hb-root]");
if (root) init(root);

function init(root: HTMLElement) {
  root.classList.add("hb-js");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── Tabs (WAI-ARIA tabs pattern, automatic activation) ──
  root.querySelectorAll<HTMLElement>("[data-hb-tabs]").forEach((group) => {
    const tabs = Array.from(group.querySelectorAll<HTMLButtonElement>("[role=tab]"));
    const panels = Array.from(group.querySelectorAll<HTMLElement>("[data-hb-panel]"));
    const vertical = group.dataset.hbOrientation === "vertical";
    const select = (i: number, focus = false) => {
      tabs.forEach((t, j) => {
        const on = i === j;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
      });
      panels.forEach((p, j) => {
        p.hidden = i !== j;
        if (i === j && !reduceMotion) {
          p.classList.remove("hb-panel-in");
          void p.offsetWidth; // restart the fade
          p.classList.add("hb-panel-in");
        }
      });
      if (focus) tabs[i].focus();
    };
    tabs.forEach((t, i) => {
      t.addEventListener("click", () => select(i));
      t.addEventListener("keydown", (e) => {
        const next = vertical ? "ArrowDown" : "ArrowRight";
        const prev = vertical ? "ArrowUp" : "ArrowLeft";
        let k = -1;
        if (e.key === next) k = (i + 1) % tabs.length;
        else if (e.key === prev) k = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") k = 0;
        else if (e.key === "End") k = tabs.length - 1;
        if (k >= 0) { e.preventDefault(); select(k, true); }
      });
    });
    select(Math.max(0, panels.findIndex((p) => p.dataset.initial === "1")));
  });

  // ── Sensitivity scenarios ──
  root.querySelectorAll<HTMLElement>("[data-hb-scenarios]").forEach((wrap) => {
    const buttons = Array.from(wrap.querySelectorAll<HTMLButtonElement>("[data-scenario]"));
    const note = wrap.querySelector<HTMLElement>("[data-hb-scenario-note]");
    const apply = (b: HTMLButtonElement) => {
      buttons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      const scores = JSON.parse(b.dataset.scores || "{}") as Record<string, number>;
      const lead = Object.entries(scores).sort((a, c) => c[1] - a[1])[0]?.[0];
      wrap.querySelectorAll<HTMLElement>("[data-option]").forEach((row) => {
        const k = row.dataset.option!;
        const fill = row.querySelector<HTMLElement>(".hb-bar-fill");
        const val = row.querySelector<HTMLElement>("[data-value]");
        if (fill) fill.style.width = `${(scores[k] / 5) * 100}%`;
        if (val) val.textContent = scores[k].toFixed(2);
        row.classList.toggle("is-lead", k === lead);
      });
      if (note) note.textContent = b.dataset.note || "";
      track("briefing_scenario_select", { scenario: b.dataset.scenario || "" });
    };
    buttons.forEach((b) => b.addEventListener("click", () => apply(b)));
    const first = buttons[0];
    if (first) {
      const scores = JSON.parse(first.dataset.scores || "{}") as Record<string, number>;
      const lead = Object.entries(scores).sort((a, c) => c[1] - a[1])[0]?.[0];
      wrap.querySelector<HTMLElement>(`[data-option="${lead}"]`)?.classList.add("is-lead");
    }
  });

  // ── Current chapter marker + reading progress ──
  const links = new Map<string, HTMLAnchorElement>();
  root.querySelectorAll<HTMLAnchorElement>("[data-hb-navlink]").forEach((a) => links.set(a.dataset.hbNavlink!, a));
  const chapters = Array.from(root.querySelectorAll<HTMLElement>("[data-hb-chapter]"));
  const visible = new Set<string>();
  let lastCurrent: string | undefined;
  const mark = () => {
    const current = chapters.find((c) => visible.has(c.id));
    // Back above the first chapter: return the scrolling bar to its start.
    if (!current && lastCurrent) {
      root.querySelector<HTMLElement>(".hb-nav-list")?.scrollTo({ left: 0, behavior: reduceMotion ? "auto" : "smooth" });
    }
    lastCurrent = current?.id;
    links.forEach((a, id) => {
      if (current && id === current.id) {
        if (!a.hasAttribute("aria-current")) {
          a.setAttribute("aria-current", "true");
          // Keep the active chapter visible in the horizontally scrolling bar
          // (scrolls the bar only, never the page).
          const list = a.closest<HTMLElement>(".hb-nav-list");
          if (list) list.scrollTo({ left: a.offsetLeft - list.clientWidth / 2 + a.clientWidth / 2, behavior: reduceMotion ? "auto" : "smooth" });
        }
      } else a.removeAttribute("aria-current");
    });
  };
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id)));
      mark();
    }, { rootMargin: "-40% 0px -55% 0px" });
    chapters.forEach((c) => io.observe(c));
  }
  const bar = root.querySelector<HTMLElement>("[data-hb-progress]");
  if (bar) {
    let ticking = false;
    const update = () => {
      const r = root.getBoundingClientRect();
      const total = r.height - window.innerHeight;
      const p = total > 0 ? Math.min(1, Math.max(0, -r.top / total)) : 0;
      bar.style.transform = `scaleX(${p})`;
      ticking = false;
    };
    window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    update();
  }

  // ── Source refs: jump to a document card and highlight it briefly ──
  root.querySelectorAll<HTMLAnchorElement>("[data-hb-docref]").forEach((a) => {
    a.addEventListener("click", () => {
      const target = document.querySelector<HTMLElement>(a.getAttribute("href")!);
      if (!target) return;
      target.classList.remove("is-flash");
      void target.offsetWidth;
      target.classList.add("is-flash");
    });
  });

  // ── Full-screen viewer for large screens; new tab everywhere else ──
  // The viewer never touches browser history. A fresh iframe is created with
  // its src set before insertion (an initial load adds no history entry) and
  // removed on close, so one click on Close or one Escape always closes it,
  // and the Back button keeps working normally.
  const dialog = root.querySelector<HTMLDialogElement>("[data-hb-viewer]");
  const holder = dialog?.querySelector<HTMLElement>("[data-hb-viewer-frame]");
  const title = dialog?.querySelector<HTMLElement>("[data-hb-viewer-title]");
  const newTab = dialog?.querySelector<HTMLAnchorElement>("[data-hb-viewer-newtab]");
  const canEmbed = () => window.matchMedia("(min-width: 900px) and (hover: hover)").matches && typeof HTMLDialogElement === "function";
  let opener: HTMLElement | null = null;
  if (dialog && holder && title && newTab) {
    root.querySelectorAll<HTMLAnchorElement>("[data-hb-view]").forEach((a) => {
      a.addEventListener("click", (e) => {
        if (!canEmbed() || e.metaKey || e.ctrlKey || e.shiftKey) return; // let the link open normally
        e.preventDefault();
        opener = a;
        const href = a.dataset.hbView!;
        const label = a.dataset.hbViewTitle || "Document";
        const frame = document.createElement("iframe");
        frame.title = label;
        frame.src = href;
        holder.replaceChildren(frame);
        title.textContent = label;
        newTab.href = href;
        dialog.showModal();
      });
    });
    dialog.querySelector("[data-hb-viewer-close]")?.addEventListener("click", () => dialog.close());
    // Escape fires "cancel" and the dialog closes itself; nothing to add.
    dialog.addEventListener("close", () => { holder.replaceChildren(); opener?.focus(); });
  }

  // ── Reveal on scroll (skipped entirely for reduced motion) ──
  const reveals = root.querySelectorAll<HTMLElement>(".hb-reveal");
  if (!reduceMotion && "IntersectionObserver" in window) {
    root.classList.add("hb-motion");
    const ro = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("is-in"); ro.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.02 });
    reveals.forEach((el) => {
      // Anything already on screen (or above it, after a deep link) shows at once.
      if (el.getBoundingClientRect().top < window.innerHeight) el.classList.add("is-in");
      else ro.observe(el);
    });
  }
}

function track(name: string, params: Record<string, string>) {
  const w = window as unknown as { gtag?: (...a: unknown[]) => void };
  try { if (typeof w.gtag === "function") w.gtag("event", name, params); } catch { /* never break the page */ }
}
