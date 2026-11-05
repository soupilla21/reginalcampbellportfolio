// Behavior for /advisory/first-90-days. Progressive enhancement only: every
// panel, row and artifact is in the HTML and readable without this script.
// Adds the Steering 2 stepper (WAI-ARIA tabs), the role picker, the current
// section marker, hash-to-card opening and restrained reveal effects.

const root = document.querySelector<HTMLElement>("[data-gp-root]");
if (root) init(root);

function init(root: HTMLElement) {
  root.classList.add("gp-js");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── Steering 2 stepper ──
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

  // ── Role picker: highlights the role's row and marks its artifacts ──
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
        const name = buttons.find((b) => b.dataset.role === role)?.textContent ?? "";
        status.textContent = all ? "" : `${name}: you'd receive ${count} of the 12 artifacts below. They're marked "You receive this."`;
      }
    };
    buttons.forEach((b) => b.addEventListener("click", () => apply(b.dataset.role!)));
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
