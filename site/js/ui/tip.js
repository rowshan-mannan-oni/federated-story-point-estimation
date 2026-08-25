/* ==========================================================================
   tip.js — the one popup the site uses.

   Both the number-provenance chips and the glossary terms need "show a short
   explanation next to this word". Two systems would drift apart, so there is
   one, and it follows three rules:

     1. It opens on hover AND on keyboard focus. A definition only available
        to a mouse is a definition half the readers cannot get to.
     2. It is attached to <body>, not next to the word. The stops sit inside
        a transformed track, and a transformed ancestor makes `position:fixed`
        resolve against that ancestor instead of the window — the classic way
        popups end up in the wrong place. Attaching to body avoids it.
     3. It never traps anything. Escape, blur, scroll or a click elsewhere
        dismisses it.
   ========================================================================== */

let tipEl = null;
let owner = null;
let hideTimer = 0;

function ensure() {
  if (tipEl) return tipEl;
  tipEl = document.createElement("div");
  tipEl.className = "tip mat-panel";
  tipEl.setAttribute("role", "tooltip");
  tipEl.id = "site-tip";
  tipEl.hidden = true;
  document.body.append(tipEl);
  return tipEl;
}

function place(anchor) {
  const tip = ensure();
  const rect = anchor.getBoundingClientRect();
  const margin = 10;

  tip.style.visibility = "hidden";
  tip.hidden = false;
  const box = tip.getBoundingClientRect();

  // Prefer above; flip below when there is no room up there.
  let top = rect.top - box.height - margin;
  if (top < margin) top = rect.bottom + margin;

  // Keep it on screen horizontally, centred on the word where possible.
  let left = rect.left + rect.width / 2 - box.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - box.width - margin));

  tip.style.top = `${Math.round(top)}px`;
  tip.style.left = `${Math.round(left)}px`;
  tip.style.visibility = "";
}

/** Show `html` next to `anchor`. Content is built by the caller, as nodes. */
export function show(anchor, content) {
  window.clearTimeout(hideTimer);
  const tip = ensure();

  tip.replaceChildren(content);
  owner = anchor;
  anchor.setAttribute("aria-describedby", tip.id);
  place(anchor);
  tip.dataset.open = "true";
}

export function hide() {
  if (!tipEl) return;
  owner?.removeAttribute("aria-describedby");
  owner = null;
  tipEl.dataset.open = "false";
  hideTimer = window.setTimeout(() => { if (tipEl) tipEl.hidden = true; }, 120);
}

/**
 * Wire an element so it shows a tip. `build()` is called each time and must
 * return a DOM node — never a string, so nothing can inject markup.
 */
export function attach(anchor, build) {
  const open = () => show(anchor, build());
  anchor.addEventListener("mouseenter", open);
  anchor.addEventListener("focus", open);
  anchor.addEventListener("mouseleave", hide);
  anchor.addEventListener("blur", hide);
  anchor.addEventListener("click", (event) => {
    event.preventDefault();
    if (tipEl?.dataset.open === "true" && owner === anchor) hide();
    else open();
  });
}

/* Global dismissals. */
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hide();
});
window.addEventListener("scroll", hide, true);
window.addEventListener("resize", hide);
