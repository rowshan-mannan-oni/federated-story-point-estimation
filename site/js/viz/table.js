/* ==========================================================================
   table.js — a table you can sort, and read out loud.

   Several stops need "here are the eighteen projects / eight approaches, with
   four numbers each". Written once here so they all sort the same way, mark
   the sorted column the same way, and stay readable on a narrow screen.

   Sorting is done with a real <button> inside the header cell and an
   aria-sort attribute on the cell, which is what screen readers actually
   listen for. A table that can only be sorted by mouse is a table half the
   readers cannot use.
   ========================================================================== */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * @param {object} options
 *   columns  [{ key, label, align, format(value,row), sortable, width }]
 *   rows     array of plain objects
 *   sort     { key, dir: "asc"|"desc" } to start with
 *   onRow    optional (row, tr) => void, for per-row decoration
 *   caption  a visually hidden description of what the table holds
 */
export function createTable({ columns = [], rows = [], sort, onRow, caption } = {}) {
  const table = el("table", "dtable");
  let order = sort ? { ...sort } : null;

  if (caption) {
    const cap = el("caption", "sr-only", caption);
    table.append(cap);
  }

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");

  const heads = columns.map((column) => {
    const th = document.createElement("th");
    th.scope = "col";
    if (column.align) th.dataset.align = column.align;
    if (column.width) th.style.setProperty("width", column.width);

    if (column.sortable === false) {
      th.append(el("span", null, column.label));
    } else {
      const button = el("button", "dtable-sort");
      button.type = "button";
      button.append(el("span", null, column.label));
      button.append(el("i", "dtable-arrow"));
      button.addEventListener("click", () => {
        const same = order?.key === column.key;
        order = { key: column.key, dir: same && order.dir === "desc" ? "asc" : "desc" };
        draw();
      });
      th.append(button);
    }
    headRow.append(th);
    return th;
  });

  thead.append(headRow);
  table.append(thead);

  const body = document.createElement("tbody");
  table.append(body);

  function draw() {
    // Mark the sorted column for assistive technology as well as the eye.
    columns.forEach((column, i) => {
      const active = order?.key === column.key;
      heads[i].setAttribute("aria-sort",
        active ? (order.dir === "asc" ? "ascending" : "descending") : "none");
      heads[i].dataset.sorted = String(active);
      heads[i].dataset.dir = active ? order.dir : "";
    });

    let list = [...rows];
    if (order) {
      const { key, dir } = order;
      list.sort((a, b) => {
        const x = a[key];
        const y = b[key];
        const cmp = typeof x === "number" && typeof y === "number"
          ? x - y
          : String(x).localeCompare(String(y));
        return dir === "asc" ? cmp : -cmp;
      });
    }

    body.replaceChildren();
    for (const row of list) {
      const tr = document.createElement("tr");
      for (const column of columns) {
        const td = document.createElement("td");
        if (column.align) td.dataset.align = column.align;
        const value = row[column.key];
        const rendered = column.format ? column.format(value, row) : value;
        if (rendered instanceof Object && rendered.nodeType) td.append(rendered);
        else td.textContent = rendered == null ? "—" : String(rendered);
        tr.append(td);
      }
      onRow?.(row, tr);
      body.append(tr);
    }
  }

  draw();

  return {
    el: table,
    redraw: draw,
    sortBy(key, dir = "desc") { order = { key, dir }; draw(); },
  };
}
