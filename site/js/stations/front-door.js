/* ==========================================================================
   Stop 1 — The front door.

   Every stop is a module with the same shape: it is handed an empty element
   and fills it. Numbers are written as <span data-fact="..."> and terms as
   <span data-term="...">; app.js swaps them for the real thing after mounting,
   so a stop never has to know where a figure came from — only which one it
   wants.

   Build step 5 will deepen this into the full opening: what the project is in
   six sentences, and the three questions it sets out to answer.
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
        Software teams estimate how much work a task will take, in
        <span data-term="story point">story points</span>. They are not very good at it.
        This thesis asks whether a model can learn to estimate from the text of the task
        itself &mdash; and whether separate projects can teach that model
        <span data-term="federated learning">together</span>, without any of them
        handing over their data.
      </p>

      <hr class="rule" />

      <p class="note">
        The evidence is real: <span data-fact="corpus.projects" data-fact-label="projects"></span>
        from public issue trackers, <span data-fact="corpus.rows_raw" data-fact-label="issues"></span>
        between <span data-fact="corpus.first_issue"></span> and
        <span data-fact="corpus.last_issue"></span>, of which
        <span data-fact="split.clients" data-fact-label="projects"></span> train together
        and one is held back to give the model a head start.
      </p>

      <p class="note">
        Hover any number on this site &mdash; or tab to it &mdash; and it will tell you
        which file it came from and how it was worked out. Underlined words are
        explained the same way. Nothing here is typed in by hand.
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
          <span class="v">Jump anywhere; solid ticks are built, hollow ones are not yet</span>
        </li>
        <li class="how-item mat-sub">
          <span class="k">The address bar</span>
          <span class="v">Every stop has its own link you can share or bookmark</span>
        </li>
      </ul>

      <p class="note reference-line">
        Reference pages, outside the walk-through:
        <a href="facts.html">every number the site knows</a> ·
        <a href="parts.html">the parts these stops are built from</a>
      </p>
    </article>
  `;
}
