// Exercise the editor's real event handlers without a browser or canvas renderer.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

class Element {
  constructor() {
    this.children = []; this.dataset = {}; this.listeners = {}; this.disabled = false;
  }
  append(...children) { this.children.push(...children); }
  add(option) { this.append(option); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this[key] = value; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  getBoundingClientRect() { return {width: 0}; }
  descendants() { return this.children.flatMap(child => [child, ...child.descendants()]); }
  querySelectorAll(selector) {
    const [, name, value] = selector.match(/^\[data-([\w-]+)(?:="([^"]*)")?\]$/);
    const key = name.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    return this.descendants().filter(child => key in child.dataset &&
      (value === undefined || child.dataset[key] === value));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

const root = new Element();
const document = {
  createElement: () => new Element(),
  createElementNS: () => new Element(),
  createTextNode: () => new Element(),
  getElementById: id => root.descendants().find(child => child.id === id) || null
};
for (const id of ["box-canvas", "reset-view", "monomer-legend", "model-summary", "molecules", "box"]) {
  const element = new Element(); element.id = id; root.append(element);
}
const context = vm.createContext({document,
  Option: class extends Element {
    constructor(text, value) { super(); this.textContent = text; this.value = String(value); }
  },
  ResizeObserver: class { observe() {} }
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../../scripts/gui/assets/model-editor.js"), "utf8"), context);
const editor = vm.runInContext("ModelEditor", context);
let changes = 0;
editor.init(document.getElementById("molecules"), document.getElementById("box"), () => changes++);
editor.loadPreset({mode: "SCFT", nx: "8, 8, 8", lx: "3, 3, 3", b_a: "1", b_b: "1", blocks: "A:0.5,B:0.5", chi_n: "12"});

const model = () => JSON.parse(JSON.stringify(editor.serialize()));
const element = id => { const result = document.getElementById(id); assert.ok(result, id); return result; };
const change = (id, value, event = "change") => {
  const target = element(id);
  assert.equal(target.disabled, false, id);
  const before = changes;
  target.value = String(value); target.listeners[event]();
  assert.equal(changes, before + 1, "Every edit must notify the script preview");
};
const options = () => element("box-shape").children.map(option => option.value);
const exportedModels = [];
for (const [dimension, shape, system, lengths, angles, free] of [
  [3, "tetragonal", "Tetragonal", ["7/2", "7/2", "5"], ["90", "90", "90"], []],
  [3, "monoclinic", "Monoclinic", ["7/2", "4", "5"], ["90", "105", "90"], [1]],
  [2, "hexagonal", "Hexagonal2D", ["7/2", "7/2"], ["90", "90", "120"], []],
  [2, "oblique", "Oblique2D", ["7/2", "4"], ["90", "90", "75"], [2]]
]) {
  change("box-dimension", dimension);
  assert.ok(options().includes(shape));
  change("box-shape", shape);
  change("box-lx-0", lengths[0], "input");
  const tied = shape === "tetragonal" || shape === "hexagonal";
  assert.equal(element("box-lx-1").disabled, tied);
  if (!tied) change("box-lx-1", lengths[1], "input");
  if (dimension === 3) change("box-lx-2", lengths[2], "input");
  for (let i = 0; i < 3; i++) {
    assert.equal(Boolean(document.getElementById(`box-angle-${i}`)), free.includes(i));
    if (free.includes(i)) change(`box-angle-${i}`, angles[i], "input");
  }
  assert.deepEqual(model().box.lx, lengths);
  assert.equal(element("box-lx-1").value, lengths[1]);
  assert.deepEqual(model().box.angles, angles);
  exportedModels.push({model: model(), system});
  for (const mode of ["L-FTS", "CL-FTS"]) {
    editor.setMode(mode);
    assert.equal(model().box.dimension, 3);
    assert.equal(element("box-dimension").disabled, true);
    assert.deepEqual(options(), ["cubic", "orthorhombic"]);
    assert.deepEqual(model().box.angles, ["90", "90", "90"]);
    editor.setMode("SCFT");
    change("box-dimension", dimension);
    change("box-shape", shape);
  }
  change("box-dimension", 1);
  assert.deepEqual(model().box.angles, ["90", "90", "90"]);
  assert.deepEqual(options(), ["line"]);
}
process.stdout.write(JSON.stringify(exportedModels));
