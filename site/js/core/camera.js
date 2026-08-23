/* ==========================================================================
   camera.js — moving between stops.

   The stops sit side by side on one long track. Moving to a stop slides the
   track so that stop fills the view; the floor behind it drifts more slowly,
   which is what makes it read as travelling rather than as a slideshow.

   The camera does not decide anything. It is told an index and it goes there.
   Deciding is the router's job.
   ========================================================================== */

let track = null;
let backdrop = null;
let current = 0;

export function init({ trackEl, backdropEl }) {
  track = trackEl;
  backdrop = backdropEl;
}

/**
 * Move the view to `index`.
 * `animate: false` jumps with no transition — used on first load and on
 * back/forward, where a slide would be a lie about what just happened.
 */
export function go(index, { animate = true } = {}) {
  if (!track) return;
  current = index;

  const shift = `translate3d(${-index * 100}%, 0, 0)`;

  if (animate) {
    track.style.transform = shift;
  } else {
    // Turn the transition off, move, then force the browser to apply the
    // change before turning it back on — otherwise the "off" never lands.
    const previous = track.style.transition;
    track.style.transition = "none";
    track.style.transform = shift;
    void track.offsetWidth;                 // read forces a style flush
    track.style.transition = previous;
  }

  // The floor drifts a fraction of the distance: parallax, so depth is felt.
  if (backdrop) backdrop.style.setProperty("--cam", String(index));
}

export function currentIndex() {
  return current;
}
