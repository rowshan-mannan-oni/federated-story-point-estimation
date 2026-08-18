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

## What's here so far

| Stop | What it covers | State |
|---|---|---|
| 1 | The room — surfaces, lighting, the toolbar, power-on | built |
| 2–29 | The walk-through itself | being built, one stop at a time |

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
index.html            the room
styles/tokens.css     greys, spacing, type, and the depth ladder — everything starts here
styles/materials.css  what things are made of: panel, recess, glass, lamps
styles/spatial.css    the shell, the view, the backdrop, and the sense of depth
styles/components.css the fittings: the two bars, buttons, tags, the power-on
js/core/store.js      the reader's three settings, remembered between visits
js/boot.js            powers the room on and wires the toolbar
```

Four rules the rest of the build follows:

1. **Greyscale only.** No hue is introduced anywhere, for any reason.
2. **One light source, from above.** Highlights on top edges, shadows below.
3. **Layout, not props.** Fixed bands that reserve their own space; sizes in `rem` and
   `ch`. Nothing may be positioned by guesswork, because that is what breaks on a
   different screen.
4. **No number is typed by hand.** From stop 4 onward every figure comes from
   `tools/extract_facts.py`, which reads the project's own data and results. Hovering a
   number shows where it came from.
