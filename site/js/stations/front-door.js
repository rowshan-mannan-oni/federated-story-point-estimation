/* ==========================================================================
   Stop 1 — The front door.

   Every stop is a module with the same shape: it is handed an empty element
   and fills it. That keeps stops independent, and lets the site load a stop's
   content only when someone actually walks to it.

   This one is deliberately short. Build step 5 will deepen it into the full
   opening: what the project is in six sentences, and the three questions it
   sets out to answer.
   ========================================================================== */

export function mount(el) {
  el.innerHTML = `
    <article class="card mat-panel">
      <p class="overline label">
        <span>FedSP-PEFT</span>
        <span class="bar" aria-hidden="true"></span>
        <span>Start here</span>
      </p>

      <h1>Guessing effort, without sharing the evidence</h1>

      <p class="lead">
        Software teams estimate how much work a task will take. They are not very
        good at it. This thesis asks whether a model can learn to estimate from the
        text of the task itself &mdash; and whether nineteen separate projects can
        teach that model together <em>without any of them handing over their data</em>.
      </p>

      <hr class="rule" />

      <p class="note">
        This is a walk-through, not a report. Each stop answers the question the
        stop before it ran into, so you can start here and keep going.
      </p>

      <ul class="how">
        <li class="how-item mat-sub">
          <span class="k"><kbd>&rarr;</kbd> <span class="sep">or</span> Next</span>
          <span class="v">Move on to the next stop</span>
        </li>
        <li class="how-item mat-sub">
          <span class="k"><kbd>M</kbd></span>
          <span class="v">See the whole path at once</span>
        </li>
        <li class="how-item mat-sub">
          <span class="k">The rail below</span>
          <span class="v">Jump anywhere; solid ticks are built, faint ones are not yet</span>
        </li>
        <li class="how-item mat-sub">
          <span class="k">The address bar</span>
          <span class="v">Every stop has its own link you can share or bookmark</span>
        </li>
      </ul>
    </article>
  `;
}
