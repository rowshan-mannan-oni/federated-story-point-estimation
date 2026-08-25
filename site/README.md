# The story point workshop

A walk-through explainer for this thesis: teaching a model to guess how much work a
software task is, from the text of the issue, **without any project handing over its
data**.

It is built as one room you move around in, not a stack of pages. You start at the front
door and follow a path; each stop solves the problem the stop before it ran into.

## Run it

The site uses JavaScript modules, and browsers only load those over `http://` — opening
`index.html` straight from disk shows a blank room. From the **project root** (the folder
above this one) run:

```
python -m http.server 8000
```

then open <http://localhost:8000/site/>.

Nothing is downloaded while the site runs: no fonts, no libraries, no trackers. It works
with the network cable pulled out.

## Getting around

The walk-through is **25 stops** long. Each one answers the question the stop before it
ran into, so the intended way through is simply forward.

| | |
|---|---|
| **Next / Back** | the two buttons at the bottom |
| **&larr; &rarr;** | same thing, from the keyboard |
| **Home / End** | first stop, last stop |
| **M** | open the map — the whole path on one screen |
| **The rail** | jump to any stop; solid ticks are built, hollow ones are not yet |

Every stop has its own address, so any of them can be linked to or bookmarked:
`http://localhost:8000/site/#/archive`. The browser's Back and Forward buttons work
normally.

## What's here so far

| Build step | What it added | State |
|---|---|---|
| 1 | The room — surfaces, light, the two bars, the three switches | done |
| 2 | Movement — the track, the router, the rail, the map | done |
| 3 | The parts bin — every control and readout the stops are built from | done |
| 4 | The facts pipeline — real numbers, provenance, stamps, glossary | done |
| 5–29 | The stops themselves | one at a time |

Stops that have no content yet say so plainly, and name the build step that will fill
them, rather than pretending to be finished or hiding from the map.

## Where the numbers come from

**No figure is ever typed into a page.** `tools/extract_facts.py` reads the project's own
data and results and writes `data/*.json`; the pages read only from there.

```
python site/tools/extract_facts.py
```

It reads `data_to_train_on/*.csv` and `results/*.json`, writes nothing outside
`site/data/`, never trains anything, and carries on with a clear note when an input is
absent. Re-run it whenever the data or the results change, and the whole site updates.

Every number on the site is a button: hover it, or tab to it, and it names the file it
came from and how it was worked out. Each is stamped with how solid it is —
`MEASURED` (recomputed here), `FROM A RUN`, `WORKED OUT` (arithmetic), `SIMULATED`
(a demonstration, never an experiment), or `NO DATA`.

`data/glossary.json` is the one hand-written file in `data/`, because definitions are
words rather than measurements. Terms marked in the prose are explained on hover.

Everything the site knows is listed on **facts.html**.

## The three switches in the toolbar

- **Light** — day, night, or follow your computer's setting.
- **Surfaces** — *real* or *flat*. Flat removes the grain, bevels and shadows and keeps
  every word, number and position identical. Use it if the texture makes text harder to
  read.
- **Motion** — *on* or *still*. Still stops every animation. If your operating system
  already asks for reduced motion, the site starts still without being told.

## The look

Greyscale only — ink, paper, and the greys between. No colour anywhere, so emphasis has
to come from contrast, weight and space. Surfaces are physical but quiet: one light from
above, a fine grain, a lit top edge, and a shadow from a fixed four-rung ladder. Nothing
is sized in viewport units and nothing is positioned by hand, which is what keeps the
layout from breaking on a different screen.

## How the files fit together

```
index.html              the room: two bars and the view between them
styles/tokens.css       greys, spacing, type, depth ladder — everything starts here
styles/materials.css    what things are made of: panel, recess, glass, lamps
styles/spatial.css      the shell, the track, a stop, the backdrop
styles/components.css   the fittings: bars, buttons, rail, map, power-on
js/app.js               the entry point: builds the track, keeps everything in step
js/boot.js              powers the room on, wires the three switches
js/core/store.js        the reader's settings, remembered between visits
js/core/stops.js        THE PATH — 25 stops in order. The spine of the whole site.
js/core/router.js       which stop you are on; the address bar is the source of truth
js/core/camera.js       slides the track, drifts the floor behind it
js/ui/rail.js           the progress rail
js/ui/map.js            the map dialog
js/stations/<id>.js     one file per stop, loaded only when you walk to it
```

To add a stop: write `js/stations/<id>.js` exporting `mount(el)`, then flip `built: true`
on that stop in `js/core/stops.js`. Nothing else needs to change — the router, rail and
map all read that one list.

## The parts bin

Open <http://localhost:8000/site/parts.html> to see every control and readout the stops
are built from, live. It is a reference page, not a stop on the path.

```
js/ui/format.js     number formatting — one place, so "1.01 MB" looks the same everywhere
js/ui/controls.js   switch · dial · slider · segmented · stepper
js/ui/readouts.js   readout · meter · gauge · stat plate · drawer
styles/parts.css    how all of them look
```

Every part shares one shape, so a stop can wire any control to any display without
either knowing about the other:

```js
const dial = createDial({ label: "Pull towards the group", min: 0, max: 0.1, step: 0.001,
                          value: 0.01, onInput: (v) => meter.set(v) });
parent.append(dial.el);
dial.get();      // read it
dial.set(0.05);  // change it — deliberately does NOT fire onInput, so wiring cannot loop
```

Two rules these follow. **Everything works from the keyboard** with the right ARIA role —
a knob that only answers to dragging is a knob half the readers of this site cannot use.
And **a control never formats its own meaning**: it reports a number, and the stop decides
what that number says.

Four rules the rest of the build follows:

1. **Greyscale only.** No hue is introduced anywhere, for any reason.
2. **One light source, from above.** Highlights on top edges, shadows below.
3. **Layout, not props.** Fixed bands that reserve their own space; sizes in `rem` and
   `ch`. Nothing may be positioned by guesswork, because that is what breaks on a
   different screen.
4. **No number is typed by hand.** From stop 4 onward every figure comes from
   `tools/extract_facts.py`, which reads the project's own data and results. Hovering a
   number shows where it came from.
