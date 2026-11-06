// Behavior for /advisory/first-90-days. Progressive enhancement only: every
// panel, row and artifact is in the HTML and readable without this script.
// Adds the delivery-path tabs (WAI-ARIA tabs), the Day 60 decision, the role picker, the current
// section marker, hash-to-card opening and restrained reveal effects.

const root = document.querySelector<HTMLElement>("[data-gp-root]");
if (root) init(root);

function init(root: HTMLElement) {
  root.classList.add("gp-js");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── Tabs (the technical delivery path) ──
  root.querySelectorAll<HTMLElement>("[data-gp-tabs]").forEach((group) => {
    const tabs = Array.from(group.querySelectorAll<HTMLButtonElement>("[role=tab]"));
    const panels = Array.from(group.querySelectorAll<HTMLElement>("[data-gp-panel]"));
    const select = (i: number, focus = false) => {
      tabs.forEach((t, j) => {
        t.setAttribute("aria-selected", String(i === j));
        t.tabIndex = i === j ? 0 : -1;
      });
      panels.forEach((p, j) => {
        p.hidden = i !== j;
        if (i === j && !reduceMotion) {
          p.classList.remove("gp-panel-in");
          void p.offsetWidth;
          p.classList.add("gp-panel-in");
        }
      });
      if (focus) tabs[i].focus();
    };
    tabs.forEach((t, i) => {
      t.addEventListener("click", () => select(i));
      t.addEventListener("keydown", (e) => {
        let k = -1;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") k = (i + 1) % tabs.length;
        else if (e.key === "ArrowLeft" || e.key === "ArrowUp") k = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") k = 0;
        else if (e.key === "End") k = tabs.length - 1;
        if (k >= 0) { e.preventDefault(); select(k, true); }
      });
    });
    select(0);
  });

  // ── Role picker: highlights the role's card and marks its artifacts ──
  const picker = root.querySelector<HTMLElement>("[data-gp-roles]");
  const status = root.querySelector<HTMLElement>("[data-gp-role-status]");
  const rows = Array.from(root.querySelectorAll<HTMLElement>("[data-role-row]"));
  const cards = Array.from(root.querySelectorAll<HTMLElement>(".gp-card[data-roles]"));
  if (picker) {
    const buttons = Array.from(picker.querySelectorAll<HTMLButtonElement>("button[data-role]"));
    const apply = (role: string) => {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.role === role)));
      const all = role === "all";
      rows.forEach((r) => {
        r.classList.toggle("is-on", !all && r.dataset.roleRow === role);
        r.classList.toggle("is-dim", !all && r.dataset.roleRow !== role);
      });
      let count = 0;
      cards.forEach((c) => {
        const mine = !all && (c.dataset.roles ?? "").split(" ").includes(role);
        if (mine) count++;
        c.classList.toggle("is-on", mine);
        c.classList.toggle("is-dim", !all && !mine);
        const badge = c.querySelector<HTMLElement>("[data-gp-badge]");
        if (badge) badge.hidden = !mine;
      });
      if (status) {
        status.replaceChildren();
        if (!all) {
          const name = buttons.find((b) => b.dataset.role === role)?.textContent ?? "";
          status.append(`${name}: ${count} of the 12 artifacts would land on your desk. `);
          const a = document.createElement("a");
          a.href = "#artifacts"; a.className = "gp-link"; a.textContent = "See them in the operating model";
          status.append(a);
        }
      }
    };
    buttons.forEach((b) => b.addEventListener("click", () => apply(b.dataset.role!)));
  }

  // ── The Day 60 decision: the visitor picks an option ──
  // Without JS every option and the recommendation are simply shown.
  const choose = root.querySelector<HTMLElement>("[data-gp-choose]");
  if (choose) {
    const choices = Array.from(choose.querySelectorAll<HTMLButtonElement>("[data-gp-choice]"));
    const panels = Array.from(choose.querySelectorAll<HTMLElement>("[data-gp-optpanel]"));
    const rec = choose.querySelector<HTMLElement>("[data-gp-rec]");
    const showRec = choose.querySelector<HTMLButtonElement>("[data-gp-show-rec]");
    const hint = choose.querySelector<HTMLElement>("[data-gp-choose-hint]");
    panels.forEach((p) => (p.hidden = true));
    if (rec) rec.hidden = true;
    const reveal = () => { if (rec) rec.hidden = false; if (showRec) showRec.hidden = true; };
    choices.forEach((c) => c.addEventListener("click", () => {
      const key = c.dataset.gpChoice;
      choices.forEach((x) => x.setAttribute("aria-pressed", String(x === c)));
      panels.forEach((p) => {
        const on = p.dataset.gpOptpanel === key;
        p.hidden = !on;
        if (on && !reduceMotion) { p.classList.remove("gp-panel-in"); void p.offsetWidth; p.classList.add("gp-panel-in"); }
      });
      if (hint) hint.hidden = true;
      reveal();
    }));
    showRec?.addEventListener("click", () => { reveal(); rec?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" }); });
    if (location.hash === "#recommendation") reveal();
  }

  // ── Opening an artifact card from a link or a deep link ──
  const openFromHash = () => {
    const id = decodeURIComponent(location.hash.slice(1));
    if (!id) return;
    const card = document.getElementById(id);
    const det = card?.querySelector("details") ?? (card instanceof HTMLDetailsElement ? card : null);
    if (det && !det.open) det.open = true;
  };
  window.addEventListener("hashchange", openFromHash);
  openFromHash();

  // ── Document reader ──
  // Large screens: the PDF opens in an on-page dialog. Phones, tablets and
  // modified clicks: the link opens the browser's own PDF viewer in a new tab.
  // A fresh iframe is created per open and removed on close, so the viewer
  // never adds browser history and Back keeps working.
  const dialog = root.querySelector<HTMLDialogElement>("[data-gp-viewer]");
  const holder = dialog?.querySelector<HTMLElement>("[data-gp-viewer-frame]");
  const vTitle = dialog?.querySelector<HTMLElement>("[data-gp-viewer-title]");
  const vTab = dialog?.querySelector<HTMLAnchorElement>("[data-gp-viewer-newtab]");
  const vDl = dialog?.querySelector<HTMLAnchorElement>("[data-gp-viewer-download]");
  const canEmbed = () => window.matchMedia("(min-width: 900px) and (hover: hover)").matches && typeof HTMLDialogElement === "function";
  let opener: HTMLElement | null = null;
  if (dialog && holder && vTitle && vTab && vDl) {
    root.querySelectorAll<HTMLAnchorElement>("[data-gp-view]").forEach((a) => {
      a.addEventListener("click", (e) => {
        if (!canEmbed() || e.metaKey || e.ctrlKey || e.shiftKey) return;
        e.preventDefault();
        opener = a.classList.contains("gp-doc-thumb") ? a.closest(".gp-doc")?.querySelector<HTMLElement>("[data-gp-view].gp-btn") ?? a : a;
        const href = a.dataset.gpView!;
        const label = a.dataset.gpViewTitle || "Document";
        const frame = document.createElement("iframe");
        frame.title = label;
        frame.src = href;
        holder.replaceChildren(frame);
        vTitle.textContent = label;
        vTab.href = href;
        vDl.href = href;
        dialog.showModal();
      });
    });
    dialog.querySelector("[data-gp-viewer-close]")?.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); }); // backdrop click
    dialog.addEventListener("close", () => { holder.replaceChildren(); opener?.focus(); });
  }

  // ── Current section marker ──
  const links = Array.from(root.querySelectorAll<HTMLAnchorElement>("[data-gp-navlink]"));
  const sections = Array.from(root.querySelectorAll<HTMLElement>("[data-gp-section]"));
  if ("IntersectionObserver" in window && sections.length) {
    const visible = new Set<string>();
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id)));
      const current = sections.find((s) => visible.has(s.id));
      links.forEach((a) => {
        if (current && a.dataset.gpNavlink === current.id) {
          if (!a.hasAttribute("aria-current")) {
            a.setAttribute("aria-current", "true");
            const list = a.closest("ol");
            if (list) list.scrollTo({ left: a.offsetLeft - 16, behavior: reduceMotion ? "auto" : "smooth" });
          }
        } else a.removeAttribute("aria-current");
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach((s) => io.observe(s));
  }

  // ── Reveal ──
  if (!reduceMotion && "IntersectionObserver" in window) {
    root.classList.add("gp-motion");
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("is-in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px" });
    root.querySelectorAll(".gp-reveal").forEach((el) => io.observe(el));
    // Anything already on screen or jumped to by hash shows at once.
    requestAnimationFrame(() => root.querySelectorAll(".gp-reveal").forEach((el) => {
      if (el.getBoundingClientRect().top < window.innerHeight) el.classList.add("is-in");
    }));
  }
}
