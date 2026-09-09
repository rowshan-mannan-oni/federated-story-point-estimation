/* ==========================================================================
   dom.js — a DOM small enough to read, big enough to test the site with.

   The stops are plain DOM code, so they can be exercised in Node without a
   browser if Node is given just enough of a DOM to work against. That is what
   this is: about two hundred lines standing in for the parts the site
   actually uses.

   It is deliberately faithful where faithfulness matters — elements carry a
   nodeType, style has setProperty, dataset maps to data- attributes — because
   every place it lied, a test passed that should not have.

   Known limitation, relied on by the suites: innerHTML is STORED, not parsed.
   A test that needs to see inside markup written that way reads the stored
   string (see deepText in the suites) rather than querying for elements.
   ========================================================================== */

class Text {
  constructor(value) {
    this._t = String(value);
    this.parentNode = null;
    this.nodeType = 3;
  }
  get textContent() { return this._t; }
}

class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.nodeType = 1;              // real elements have this; site code checks it
    this.attrs = new Map();
    this.children = [];
    this.parentNode = null;
    this._text = "";
    this.hidden = false;
    this.listeners = {};
    this.__cls = new Set();

    this.style = {
      _props: {},
      setProperty(key, value) { this._props[key] = String(value); },
      getPropertyValue(key) { return this._props[key] ?? ""; },
      removeProperty(key) { delete this._props[key]; },
    };

    const toAttr = (key) =>
      "data-" + String(key).replace(/[A-Z]/g, (m) => "-" + m.toLowerCase());
    this.dataset = new Proxy({}, {
      get: (_, key) => this.attrs.get(toAttr(key)),
      set: (_, key, value) => { this.attrs.set(toAttr(key), String(value)); return true; },
      has: (_, key) => this.attrs.has(toAttr(key)),
      deleteProperty: (_, key) => { this.attrs.delete(toAttr(key)); return true; },
    });

    this.classList = {
      add: (c) => { this.__cls.add(c); this._sync(); },
      remove: (c) => { this.__cls.delete(c); this._sync(); },
      contains: (c) => this.__cls.has(c),
      toggle: (c, on) => { on ? this.__cls.add(c) : this.__cls.delete(c); this._sync(); },
    };
  }

  _sync() { this.attrs.set("class", [...this.__cls].join(" ")); }

  get className() { return this.attrs.get("class") || ""; }
  set className(value) {
    this.attrs.set("class", value);
    this.__cls = new Set(String(value).split(/\s+/).filter(Boolean));
  }

  get id() { return this.attrs.get("id") || ""; }
  set id(value) { this.attrs.set("id", value); }

  get textContent() {
    return this.children.length
      ? this.children.map((c) => c.textContent).join("")
      : this._text;
  }
  set textContent(value) { this._text = String(value); this.children = []; }

  /* Stored, not parsed — see the note at the top of this file. */
  set innerHTML(value) { this._html = value; }
  get innerHTML() { return this._html || ""; }

  setAttribute(key, value) { this.attrs.set(key, String(value)); }
  getAttribute(key) { return this.attrs.has(key) ? this.attrs.get(key) : null; }
  removeAttribute(key) { this.attrs.delete(key); }
  hasAttribute(key) { return this.attrs.has(key); }

  append(...nodes) {
    for (const raw of nodes) {
      const node = typeof raw === "string" ? new Text(raw) : raw;
      node.parentNode = this;
      this.children.push(node);
    }
  }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  replaceWith(node) {
    const parent = this.parentNode;
    if (!parent) return;
    parent.children[parent.children.indexOf(this)] = node;
    node.parentNode = parent;
  }
  remove() {
    const parent = this.parentNode;
    if (!parent) return;
    parent.children = parent.children.filter((c) => c !== this);
    this.parentNode = null;
  }

  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn);
  }
  /** Test-only: fire a listener directly. */
  fire(type, event = {}) {
    (this.listeners[type] || []).forEach((fn) =>
      fn({ preventDefault() {}, stopPropagation() {}, target: this, ...event }));
  }
  focus() {}

  getBoundingClientRect() {
    return { top: 100, left: 100, width: 80, height: 20, bottom: 120, right: 180 };
  }

  _all(out = []) {
    for (const child of this.children) {
      if (child instanceof El) { out.push(child); child._all(out); }
    }
    return out;
  }

  matches(selector) {
    if (selector.startsWith("[")) {
      const m = selector.match(/^\[([^\]=]+)(?:="([^"]*)")?\]$/);
      if (!m) return false;
      return m[2] === undefined ? this.attrs.has(m[1]) : this.attrs.get(m[1]) === m[2];
    }
    if (selector.startsWith(".")) return this.__cls.has(selector.slice(1));
    return this.tagName === selector.toUpperCase();
  }

  querySelectorAll(selector) {
    const parts = selector.split(",").map((s) => s.trim());
    return this._all().filter((e) => parts.some((s) => e.matches(s)));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
}

const document = {
  createElement: (tag) => new El(tag),
  createTextNode: (text) => new Text(text),
  body: new El("body"),
  documentElement: new El("html"),
};

const window = {
  addEventListener() {},
  removeEventListener() {},
  innerWidth: 1200,
  innerHeight: 800,
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (id) => clearTimeout(id),
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
};

export function install() {
  globalThis.document = document;
  globalThis.window = window;
  globalThis.Text = Text;
  globalThis.localStorage = window.localStorage;
  // Readouts animate their value with rAF, re-scheduling until the duration
  // has elapsed. Handing the callback a timestamp far in the future finishes
  // the animation in a single frame, so a test sees the settled number and
  // the callback never re-schedules into an unbounded recursion.
  globalThis.requestAnimationFrame = (fn) => { fn(performance.now() + 1e9); return 0; };
  globalThis.cancelAnimationFrame = () => {};
}

/**
 * Everything a card renders, as one whitespace-collapsed string.
 *
 * Reads stored innerHTML as well as text nodes, because this shim does not
 * parse markup. Whitespace is collapsed so that a line break inside a template
 * literal — which HTML would not render as a difference — is not one here.
 */
export function deepText(node, out = []) {
  if (node.innerHTML) out.push(node.innerHTML);
  if (node._text) out.push(node._text);
  for (const child of node.children || []) deepText(child, out);
  return out.join(" ").replace(/\s+/g, " ");
}

export { El, Text, document, window };
