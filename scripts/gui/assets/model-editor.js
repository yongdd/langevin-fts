"use strict";

const ModelEditor = (() => {
  const palette = ["#238474", "#d49a46", "#7771b9", "#cb6a73", "#518bb3", "#9b9e46", "#b076a5", "#66895e"];
  const shapes = {1: [["line", "Line"]],
    2: [["square", "Square"], ["rectangle", "Rectangle"], ["hexagonal", "Hexagonal"], ["oblique", "Oblique"]],
    3: [["cubic", "Cubic"], ["tetragonal", "Tetragonal"], ["orthorhombic", "Orthorhombic"],
      ["hexagonal", "Hexagonal"], ["monoclinic", "Monoclinic"], ["triclinic", "Triclinic"]]};
  let model, moleculesPanel, boxPanel, notify, serial = 0, uniform = true, mode = "SCFT";
  let yaw = .65, pitch = -.45, drag = null;
  const uid = () => `row-${++serial}`;
  const byId = id => model.monomers.find(m => m.id === id);
  const color = id => palette[Math.max(0, model.monomers.findIndex(m => m.id === id)) % palette.length];
  const numeric = value => {
    const text = String(value).trim();
    if (!text) return NaN;
    if (/^[+-]?\d+\s*\/\s*[+-]?\d+$/.test(text)) {
      const [a, b] = text.split("/").map(Number);
      return b ? a / b : NaN;
    }
    return Number(text);
  };

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function button(text, action, label = text) {
    const element = node("button", "row-button", text);
    element.type = "button";
    element.title = label;
    element.setAttribute("aria-label", label);
    element.addEventListener("click", action);
    return element;
  }
  function input(value, label, update, id) {
    const element = node("input");
    element.type = "text";
    element.value = value;
    element.maxLength = 128;
    element.spellcheck = false;
    element.setAttribute("aria-label", label);
    if (id) element.id = id;
    element.addEventListener("input", () => { update(element.value); notify(); });
    return element;
  }
  function selectMonomer(value, label, update) {
    const element = node("select");
    element.dataset.monomerSelect = "true";
    element.setAttribute("aria-label", label);
    model.monomers.forEach(m => element.add(new Option(m.name || "Unnamed", m.id)));
    element.value = value;
    element.addEventListener("change", () => { update(element.value); notify(); });
    return element;
  }
  function section(parent, title, help, actionLabel, action, id) {
    const container = node("section", "model-section");
    const header = node("div", "model-section-heading");
    header.append(node("h3", "", title));
    if (action) {
      const add = button(actionLabel, action);
      add.id = id;
      header.append(add);
    }
    container.append(header, node("p", "section-help", help));
    parent.append(container);
    return container;
  }
  function structuralChange() { renderMolecules(); notify(); }

  function nodeIds(polymer) {
    return [...new Set(polymer.blocks.flatMap(b => [b.v, b.u]))].sort((a, b) => a - b);
  }
  function nodeDegrees(polymer) {
    const degrees = new Map(nodeIds(polymer).map(id => [id, 0]));
    polymer.blocks.forEach(b => { degrees.set(b.v, degrees.get(b.v) + 1); degrees.set(b.u, degrees.get(b.u) + 1); });
    return degrees;
  }
  function attachmentNode(polymer) {
    const degrees = nodeDegrees(polymer);
    if (!degrees.has(polymer.attachNode)) {
      polymer.attachNode = [...degrees.keys()].sort((a, b) => degrees.get(b) - degrees.get(a) || a - b)[0];
    }
    return polymer.attachNode;
  }
  function nodePicker(polymer, value, label, update, allowNew = false, compact = false) {
    const select = node("select");
    select.setAttribute("aria-label", label);
    const degrees = nodeDegrees(polymer), ids = [...degrees.keys()];
    ids.forEach(id => {
      const title = `Node ${id}${degrees.get(id) > 2 ? " · junction" : ""}`;
      const option = new Option(compact ? String(id) : title, id);
      option.title = title;
      select.add(option);
    });
    if (allowNew) select.add(new Option(`+ New node (${Math.max(...ids) + 1})`, "new"));
    select.value = value;
    if (compact) select.title = `${label}: Node ${value}${degrees.get(value) > 2 ? " · junction" : ""}`;
    select.addEventListener("change", () => {
      update(select.value === "new" ? Math.max(...ids) + 1 : Number(select.value));
    });
    return select;
  }

  function treeLayout(polymer) {
    const lengths = polymer.blocks.map(b => numeric(b.length));
    if (!lengths.every(n => Number.isFinite(n) && n > 0) || !Number.isFinite(lengths.reduce((a, b) => a + b, 0))) {
      return {error: "Enter positive, finite block lengths to preview the polymer."};
    }
    const ids = nodeIds(polymer), adjacency = new Map(ids.map(id => [id, []])), pairs = new Set();
    for (let i = 0; i < polymer.blocks.length; i++) {
      const b = polymer.blocks[i], key = [b.v, b.u].sort((a, b) => a - b).join(",");
      if (b.v === b.u) return {error: "Each block must connect two different nodes."};
      if (pairs.has(key)) return {error: "Remove duplicate connections between the same nodes."};
      pairs.add(key);
      adjacency.get(b.v).push({id: b.u, index: i}); adjacency.get(b.u).push({id: b.v, index: i});
    }
    adjacency.forEach(neighbors => neighbors.sort((a, b) => a.id - b.id));
    const root = [...ids].sort((a, b) => adjacency.get(b).length - adjacency.get(a).length || a - b)[0];
    const visited = new Set(), leaves = new Map();
    let cycle = false;
    const count = (id, parent) => {
      if (visited.has(id)) { cycle = true; return 0; }
      visited.add(id);
      let total = 0;
      for (const next of adjacency.get(id)) if (next.id !== parent) total += count(next.id, id);
      leaves.set(id, total || 1); return total || 1;
    };
    count(root, null);
    if (cycle) return {error: "Rings are not supported. Connect the blocks as a tree."};
    if (visited.size !== ids.length) return {error: "Connect all blocks into one polymer to preview its architecture."};
    const positions = new Map([[root, [0, 0]]]);
    const place = (id, parent, start, span) => {
      const children = adjacency.get(id).filter(next => next.id !== parent);
      const total = children.reduce((sum, next) => sum + leaves.get(next.id), 0);
      let angle = start;
      children.forEach(next => {
        const wedge = span * leaves.get(next.id) / total, direction = angle + wedge / 2;
        const here = positions.get(id), length = lengths[next.index] / Math.max(...lengths);
        positions.set(next.id, [here[0] + length * Math.cos(direction), here[1] + length * Math.sin(direction)]);
        place(next.id, id, angle, wedge); angle += wedge;
      });
    };
    place(root, null, -Math.PI / 2, 2 * Math.PI);
    const points = [...positions.values()];
    const min = [0, 1].map(axis => Math.min(...points.map(p => p[axis])));
    const span = [0, 1].map(axis => Math.max(...points.map(p => p[axis])) - min[axis]);
    const scale = Math.max(...span, 1e-12);
    positions.forEach((p, id) => positions.set(id, p.map((x, axis) => .5 + (x - min[axis] - span[axis] / 2) / scale)));
    return {positions, adjacency};
  }

  function renderMolecules() {
    moleculesPanel.replaceChildren();
    const monomers = section(moleculesPanel, "Monomer types", "Define a name and statistical segment length for each monomer.", "+ Add monomer", () => {
      if (model.monomers.length >= 32) return;
      let index = 0, name;
      // A ... Z, then AA, AB, ...: monomer names take letters and underscores only
      do { name = (index < 26 ? "" : String.fromCharCode(64 + Math.floor(index / 26))) + String.fromCharCode(65 + index % 26); index++; }
      while (model.monomers.some(m => m.name === name));
      model.monomers.push({id: uid(), name, segment_length: "1.0"});
      structuralChange();
    }, "add-monomer");
    const monomerHeader = node("div", "edit-row monomer-row table-heading");
    monomerHeader.append(node("span", "", "Name"), node("span", "", "Segment length"), node("span"));
    monomers.append(monomerHeader);
    model.monomers.forEach((m, i) => {
      const row = node("div", "edit-row monomer-row");
      const name = input(m.name, `Monomer ${i + 1} name`, value => { m.name = value; }, `monomer-name-${i}`);
      const length = input(m.segment_length, `Monomer ${i + 1} segment length`, value => { m.segment_length = value; });
      const remove = button("×", () => { model.monomers = model.monomers.filter(item => item !== m); structuralChange(); }, `Remove monomer ${i + 1}`);
      remove.dataset.removeMonomer = m.id;
      row.append(name, length, remove);
      monomers.append(row);
    });

    const polymers = section(moleculesPanel, "Polymers", "Choose a linear chain or a branched tree. Block lengths use N_ref units and need not sum to 1.", "+ Add polymer", () => {
      if (model.polymers.length >= 64) return;
      model.polymers.push({id: uid(), architecture: "linear", volume_fraction: "0.5", blocks: model.monomers.slice(0, 2).map(m => ({id: uid(), monomer: m.id, length: "0.5"}))});
      structuralChange();
    }, "add-polymer");
    const fractionBar = node("div", "fraction-bar");
    const total = node("span"); total.id = "volume-total";
    const equal = button("Equal fractions", () => {
      model.polymers.forEach(p => { p.volume_fraction = String(1 / model.polymers.length); });
      structuralChange();
    });
    equal.id = "equal-fractions";
    fractionBar.append(total, equal);
    polymers.append(fractionBar);
    model.polymers.forEach((polymer, pi) => {
      const card = node("div", "polymer-card");
      card.dataset.polymer = polymer.id;
      const heading = node("div", "model-section-heading");
      const remove = button("Remove", () => { model.polymers = model.polymers.filter(p => p !== polymer); structuralChange(); }, `Remove polymer ${pi + 1}`);
      remove.disabled = model.polymers.length === 1;
      heading.append(node("h4", "", `Polymer ${pi + 1}`), remove);
      const volume = node("label", "volume-field", "Volume fraction");
      volume.append(input(polymer.volume_fraction, `Polymer ${pi + 1} volume fraction`, value => { polymer.volume_fraction = value; }));
      card.append(heading, volume);
      const architecture = node("select");
      architecture.setAttribute("aria-label", `Polymer ${pi + 1} architecture`);
      architecture.add(new Option("Linear", "linear")); architecture.add(new Option("Branched", "branched"));
      architecture.value = polymer.architecture;
      architecture.addEventListener("change", () => {
        polymer.architecture = architecture.value;
        polymer.blocks.forEach((b, i) => {
          if (polymer.architecture === "branched") { b.v = i; b.u = i + 1; }
          else { delete b.v; delete b.u; }
        });
        delete polymer.attachNode;
        structuralChange();
      });
      const architectureLabel = node("label", "architecture-field", "Architecture");
      architectureLabel.append(architecture);
      const branched = polymer.architecture === "branched";
      card.append(architectureLabel, node("p", "field-hint topology-help", branched ?
        "Choose the node where a new block attaches. Reuse a node to form a junction. Click a node in the diagram to select it. Rings are not supported." :
        "Blocks connect in row order. Switching to Linear reconnects all blocks in that order."));
      const degrees = branched ? nodeDegrees(polymer) : null;
      const validTree = branched && !treeLayout(polymer).error;
      const blockList = node("div", "block-list");
      blockList.setAttribute("role", "group");
      blockList.setAttribute("aria-label", `Polymer ${pi + 1} blocks`);
      const rowClass = `edit-row block-row${branched ? " branched-block-row" : ""}`;
      const header = node("div", `${rowClass} table-heading`);
      const titles = branched ? ["Type", "Length", "Start", "End", "Actions"] : ["Type", "Length", "Order / remove"];
      titles.forEach(title => header.append(node("span", "", title)));
      blockList.append(header);
      card.append(blockList);
      polymer.blocks.forEach((block, bi) => {
        const item = node("div", "block-item");
        item.dataset.block = block.id;
        const row = node("div", rowClass);
        const order = node("div", "block-actions");
        const move = offset => {
          const other = bi + offset;
          [polymer.blocks[bi], polymer.blocks[other]] = [polymer.blocks[other], polymer.blocks[bi]];
          structuralChange();
        };
        const up = button("↑", () => move(-1), `Move block ${bi + 1} up in polymer ${pi + 1}`);
        up.disabled = bi === 0;
        const down = button("↓", () => move(1), `Move block ${bi + 1} down in polymer ${pi + 1}`);
        down.disabled = bi === polymer.blocks.length - 1;
        const removeBlock = button("×", () => { polymer.blocks.splice(bi, 1); structuralChange(); }, `Remove block ${bi + 1} from polymer ${pi + 1}`);
        const interior = validTree && degrees.get(block.v) > 1 && degrees.get(block.u) > 1;
        removeBlock.disabled = polymer.blocks.length === 1 || interior;
        if (interior) removeBlock.title = "Remove terminal blocks first to keep the polymer connected";
        order.append(up, down, removeBlock);
        row.append(selectMonomer(block.monomer, `Polymer ${pi + 1} block ${bi + 1} monomer`, value => { block.monomer = value; }),
          input(block.length, `Polymer ${pi + 1} block ${bi + 1} length`, value => { block.length = value; }));
        if (branched) {
          for (const [key, title] of [["v", "Start node"], ["u", "End node"]]) {
            row.append(nodePicker(polymer, block[key], `Polymer ${pi + 1} block ${bi + 1} ${title.toLowerCase()}`, value => {
              block[key] = value; structuralChange();
            }, true, true));
          }
        }
        row.append(order);
        item.append(row);
        blockList.append(item);
      });
      const addBlock = button("+ Add block", () => {
        if (polymer.blocks.length >= 256) return;
        const block = {id: uid(), monomer: model.monomers[0].id, length: "0.5"};
        if (branched) { block.v = attachmentNode(polymer); block.u = Math.max(...nodeIds(polymer)) + 1; }
        polymer.blocks.push(block);
        structuralChange();
      }, `Add block to polymer ${pi + 1}`);
      const diagram = node("div", "architecture");
      diagram.dataset.architecture = polymer.id;
      if (branched) {
        const attach = node("label", "attach-field", "Attach new block to");
        const picker = nodePicker(polymer, attachmentNode(polymer), `Attach new block to node in polymer ${pi + 1}`, value => {
          polymer.attachNode = value; refresh();
        });
        picker.dataset.attachPolymer = polymer.id;
        attach.append(picker); card.append(attach);
      }
      card.append(addBlock, diagram);
      polymers.append(card);
    });

    const interactions = section(moleculesPanel, "Interactions", "Add a chi N value for each interacting pair. Unlisted pairs use chi N = 0.", "+ Add interaction", () => {
      const pair = availablePair();
      if (pair) model.interactions.push({id: uid(), a: pair[0], b: pair[1], chi_n: "0"});
      structuralChange();
    }, "add-interaction");
    const header = node("div", "edit-row interaction-row table-heading");
    ["Monomer 1", "Monomer 2", "χN", ""].forEach(text => header.append(node("span", "", text)));
    interactions.append(header);
    model.interactions.forEach((pair, i) => {
      const row = node("div", "edit-row interaction-row");
      row.append(selectMonomer(pair.a, `Interaction ${i + 1} first monomer`, value => { pair.a = value; }),
        selectMonomer(pair.b, `Interaction ${i + 1} second monomer`, value => { pair.b = value; }),
        input(pair.chi_n, `Interaction ${i + 1} chi N`, value => { pair.chi_n = value; }),
        button("×", () => { model.interactions = model.interactions.filter(p => p !== pair); structuralChange(); }, `Remove interaction ${i + 1}`));
      interactions.append(row);
    });
    refresh();
  }

  function availablePair() {
    for (let i = 0; i < model.monomers.length; i++) {
      for (const b of model.monomers.slice(i + 1)) {
        const a = model.monomers[i];
        if (!model.interactions.some(p => p.a === a.id && p.b === b.id || p.a === b.id && p.b === a.id)) return [a.id, b.id];
      }
    }
    return null;
  }

  function setMode(value) {
    const changed = mode !== value;
    mode = value;
    if (!model) return;
    let adjusted = false;
    if (mode !== "SCFT") {
      if (model.box.dimension !== 3) { model.box.dimension = 3; adjusted = true; }
      if (!["cubic", "orthorhombic"].includes(model.box.shape)) {
        model.box.shape = "orthorhombic"; adjusted = true;
      }
      if (model.box.angles.some(angle => numeric(angle) !== 90)) {
        model.box.angles = ["90", "90", "90"]; adjusted = true;
      }
    }
    if (changed || adjusted) renderBox();
  }

  function renderBox() {
    const fts = mode !== "SCFT";
    boxPanel.replaceChildren();
    boxPanel.append(node("p", "section-help", fts ?
      "L-FTS and CL-FTS require a 3D orthogonal box. Choose Cubic or Orthorhombic, then set grid points and lengths. Lengths use reference-chain units." :
      "Select a cell shape, then set grid points and lengths for each axis. Lengths use reference-chain units."));
    const controls = node("div", "field-grid");
    const field = (title, element) => {
      const wrapper = node("div", "field");
      const label = node("label", "", title);
      label.htmlFor = element.id;
      wrapper.append(label, element); controls.append(wrapper);
    };
    const dimensions = node("select"); dimensions.id = "box-dimension";
    (fts ? [3] : [1, 2, 3]).forEach(dim => dimensions.add(new Option(`${dim}D`, dim)));
    dimensions.value = model.box.dimension;
    dimensions.disabled = fts;
    dimensions.addEventListener("change", () => {
      model.box.dimension = Number(dimensions.value);
      model.box.shape = {1: "line", 2: "square", 3: "cubic"}[model.box.dimension];
      model.box.lx = Array(3).fill(model.box.lx[0]);
      model.box.angles = ["90", "90", "90"];
      renderBox(); notify();
    });
    const shape = node("select"); shape.id = "box-shape";
    shapes[model.box.dimension].filter(([value]) => !fts || ["cubic", "orthorhombic"].includes(value))
      .forEach(([value, title]) => shape.add(new Option(title, value)));
    shape.value = model.box.shape;
    shape.addEventListener("change", () => {
      model.box.shape = shape.value;
      if (["square", "cubic"].includes(shape.value)) model.box.lx = Array(3).fill(model.box.lx[0]);
      if (["tetragonal", "hexagonal"].includes(shape.value)) model.box.lx[1] = model.box.lx[0];
      model.box.angles = shape.value === "hexagonal" ? ["90", "90", "120"] : ["90", "90", "90"];
      renderBox(); notify();
    });
    field("Dimension", dimensions); field("Cell shape", shape);
    boxPanel.append(controls);
    const uniformLabel = node("label", "uniform-grid");
    const check = node("input"); check.type = "checkbox"; check.id = "uniform-grid"; check.checked = uniform;
    check.addEventListener("change", () => {
      uniform = check.checked;
      if (uniform) model.box.nx = Array(3).fill(model.box.nx[0]);
      renderBox(); notify();
    });
    uniformLabel.append(check, document.createTextNode("Use the same grid size on all axes"));
    boxPanel.append(uniformLabel);
    const header = node("div", "edit-row axis-row table-heading");
    ["Axis", "Grid points", "Box length"].forEach(title => header.append(node("span", "", title)));
    boxPanel.append(header);
    for (let axis = 0; axis < model.box.dimension; axis++) {
      const row = node("div", "edit-row axis-row");
      const grid = input(model.box.nx[axis], `Grid points ${"xyz"[axis]}`, value => {
        model.box.nx[axis] = value;
        if (uniform) {
          model.box.nx = Array(3).fill(value);
          boxPanel.querySelectorAll("[data-grid-axis]").forEach(element => { element.value = value; });
        }
      }, `box-nx-${axis}`);
      grid.dataset.gridAxis = axis;
      grid.disabled = uniform && axis > 0;
      const length = input(model.box.lx[axis], `Box length ${"abc"[axis]}`, value => {
        model.box.lx[axis] = value;
        const lockAll = ["square", "cubic"].includes(model.box.shape);
        if (lockAll || ["tetragonal", "hexagonal"].includes(model.box.shape) && axis === 0) {
          const count = lockAll ? 3 : 2;
          for (let i = 0; i < count; i++) {
            model.box.lx[i] = value;
            const other = document.getElementById(`box-lx-${i}`);
            if (other) other.value = value;
          }
        }
      }, `box-lx-${axis}`);
      length.disabled = axis > 0 && (["square", "cubic"].includes(model.box.shape) ||
        ["tetragonal", "hexagonal"].includes(model.box.shape) && axis === 1);
      row.append(node("span", "axis-name", "abc"[axis]), grid, length);
      boxPanel.append(row);
    }
    const freeAngles = {monoclinic: [1], triclinic: [0, 1, 2], oblique: [2]}[model.box.shape] || [];
    if (freeAngles.length) {
      const angleGrid = node("div", "angle-grid");
      freeAngles.forEach(i => {
        const name = ["Alpha (b,c)", "Beta (a,c)", "Gamma (a,b)"][i];
        const label = node("label", "", `${name} · degrees`);
        label.append(input(model.box.angles[i], name, value => { model.box.angles[i] = value; }, `box-angle-${i}`));
        angleGrid.append(label);
      });
      boxPanel.append(angleGrid);
    }
    const hint = {
      tetragonal: "Tetragonal cells use a = b and all angles fixed at 90°.",
      hexagonal: "Hexagonal cells use a = b, alpha = beta = 90° and gamma = 120°.",
      monoclinic: "Monoclinic cells use unique axis b: alpha = gamma = 90°; beta is adjustable.",
      triclinic: "All three cell angles are adjustable.",
      oblique: "The angle gamma between a and b is adjustable."
    }[model.box.shape] || "All box angles are fixed at 90°.";
    boxPanel.append(node("p", "field-hint", `${hint} All box boundaries are periodic.`));
  }

  function svg(tag, attrs = {}, text) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function branchedArchitecture(polymer, target) {
    const layout = treeLayout(polymer);
    if (layout.error) { target.append(node("p", "field-hint topology-error", layout.error)); return; }
    const {positions, adjacency} = layout;
    const drawing = svg("svg", {viewBox: "0 0 400 260", role: "group", "aria-label": "Branched polymer block architecture. Select a node to attach a new block."});
    const point = id => { const p = positions.get(id); return [35 + 330 * p[0], 25 + 210 * p[1]]; };
    polymer.blocks.forEach(block => {
      const start = point(block.v), end = point(block.u), length = Math.hypot(end[0] - start[0], end[1] - start[1]);
      let path = "";
      for (let j = 0; j <= 30; j++) {
        const t = j / 30, wiggle = Math.min(3, length / 20) * Math.sin(t * Math.PI * 8) * Math.sin(t * Math.PI);
        const x = start[0] + (end[0] - start[0]) * t - wiggle * (end[1] - start[1]) / Math.max(1, length);
        const y = start[1] + (end[1] - start[1]) * t + wiggle * (end[0] - start[0]) / Math.max(1, length);
        path += `${j ? "L" : "M"}${x},${y} `;
      }
      const edge = svg("path", {d: path, fill: "none", stroke: color(block.monomer), "stroke-width": 6, "stroke-linecap": "round", "data-edge": `${block.v},${block.u}`});
      const name = byId(block.monomer)?.name || "?";
      edge.append(svg("title", {}, `${name}: ${block.length} · nodes ${block.v}–${block.u}`));
      drawing.append(edge);
      if (polymer.blocks.length <= 24 && length > 25) {
        drawing.append(svg("text", {x: (start[0] + end[0]) / 2, y: (start[1] + end[1]) / 2 - 8,
          "text-anchor": "middle", fill: "#233632", "font-size": 10, stroke: "#fcfdfb", "stroke-width": 3, "paint-order": "stroke"},
        polymer.blocks.length <= 12 ? `${name} · ${block.length}` : name));
      }
    });
    const selected = attachmentNode(polymer);
    positions.forEach((_, id) => {
      const [x, y] = point(id), junction = adjacency.get(id).length > 2;
      const marker = svg("g", {role: "button", tabindex: "0", class: "architecture-node", "data-node-id": id,
        "aria-label": `Attach new block to node ${id}${junction ? ", junction" : ""}`, "aria-pressed": String(id === selected)});
      marker.append(svg("circle", {cx: x, cy: y, r: 8, fill: id === selected ? "#17695c" : "#fcfdfb", stroke: "#17695c", "stroke-width": 1.5}));
      marker.append(svg("text", {x, y: y + 3, "text-anchor": "middle", fill: id === selected ? "#fff" : "#233632", "font-size": 8}, id));
      const choose = () => {
        polymer.attachNode = id;
        moleculesPanel.querySelector(`[data-attach-polymer="${polymer.id}"]`).value = id;
        refresh();
        target.querySelector(`[data-node-id="${id}"]`).focus();
      };
      marker.addEventListener("click", choose);
      marker.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); }
      });
      drawing.append(marker);
    });
    const junctions = [...adjacency.values()].filter(neighbors => neighbors.length > 2).length;
    const total = polymer.blocks.reduce((sum, b) => sum + numeric(b.length), 0);
    target.append(drawing, node("p", "architecture-caption", `Branched tree · ${polymer.blocks.length} blocks · ${junctions} junction${junctions === 1 ? "" : "s"} · total length ${total.toPrecision(4)}`));
  }
  function architecture(polymer, target) {
    target.replaceChildren();
    if (polymer.architecture === "branched") { branchedArchitecture(polymer, target); return; }
    const lengths = polymer.blocks.map(b => numeric(b.length));
    if (!lengths.every(n => Number.isFinite(n) && n > 0)) {
      target.append(node("p", "field-hint", "Enter positive block lengths to preview the chain.")); return;
    }
    const sum = lengths.reduce((a, b) => a + b, 0);
    const drawing = svg("svg", {viewBox: "0 0 400 82", role: "img", "aria-label": "Linear polymer block architecture"});
    let x = 20;
    polymer.blocks.forEach((block, i) => {
      const end = x + 360 * lengths[i] / sum;
      let path = "";
      for (let j = 0; j <= 30; j++) {
        const xx = x + (end - x) * j / 30;
        path += `${j ? "L" : "M"}${xx},${42 + 5 * Math.sin((xx - 20) * .12)} `;
      }
      drawing.append(svg("path", {d: path, fill: "none", stroke: color(block.monomer), "stroke-width": 8, "stroke-linecap": "round"}));
      if (end - x > 32) {
        drawing.append(svg("text", {x: (x + end) / 2, y: 19, "text-anchor": "middle", fill: "#233632", "font-size": 12}, byId(block.monomer)?.name || "?"));
        drawing.append(svg("text", {x: (x + end) / 2, y: 73, "text-anchor": "middle", fill: "#67756e", "font-size": 10}, block.length));
      }
      x = end;
    });
    target.append(drawing, node("p", "architecture-caption", `${polymer.blocks.map(b => byId(b.monomer)?.name || "?").join(" → ")} · total length ${sum.toPrecision(4)}`));
  }

  function refresh() {
    if (!model) return;
    moleculesPanel.querySelectorAll("[data-monomer-select]").forEach(element => {
      const value = element.value;
      element.replaceChildren(...model.monomers.map(m => new Option(m.name || "Unnamed", m.id)));
      element.value = value;
    });
    moleculesPanel.querySelectorAll("[data-remove-monomer]").forEach(element => {
      const id = element.dataset.removeMonomer;
      const used = model.polymers.some(p => p.blocks.some(b => b.monomer === id)) || model.interactions.some(p => p.a === id || p.b === id);
      element.disabled = model.monomers.length <= 2 || used;
      element.title = used ? "Remove blocks and interactions using this monomer first" : "Remove monomer";
    });
    const addInteraction = document.getElementById("add-interaction");
    if (addInteraction) addInteraction.disabled = !availablePair();
    const total = document.getElementById("volume-total");
    if (total) {
      const sum = model.polymers.reduce((value, p) => value + numeric(p.volume_fraction), 0);
      total.textContent = `Total volume fraction: ${Number.isFinite(sum) ? sum.toPrecision(5) : "—"} / 1`;
      total.className = Math.abs(sum - 1) < 1e-10 ? "fraction-valid" : "fraction-invalid";
    }
    model.polymers.forEach(polymer => {
      const target = moleculesPanel.querySelector(`[data-architecture="${polymer.id}"]`);
      if (target) architecture(polymer, target);
    });
    const legend = document.getElementById("monomer-legend"); legend.replaceChildren();
    model.monomers.forEach(m => {
      const entry = node("span", "legend-entry");
      const dot = svg("svg", {viewBox: "0 0 12 12", width: 12, height: 12, "aria-hidden": "true"});
      dot.append(svg("circle", {cx: 6, cy: 6, r: 5, fill: color(m.id)}));
      entry.append(dot, document.createTextNode(m.name || "Unnamed")); legend.append(entry);
    });
    document.getElementById("model-summary").textContent = `${model.box.dimension}D · ${model.box.shape} · ${model.polymers.length} polymer${model.polymers.length === 1 ? "" : "s"}`;
    draw();
  }

  function draw() {
    if (!model) return;
    const canvas = document.getElementById("box-canvas");
    const width = canvas.getBoundingClientRect().width, height = 280;
    if (!width) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = height * dpr;
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#f5f8f4"; ctx.fillRect(0, 0, width, height);
    const box = model.box, dim = box.dimension;
    const lengths = box.lx.slice(0, dim).map(numeric), angles = box.angles.map(numeric);
    const invalid = () => {
      ctx.fillStyle = "#67756e"; ctx.font = "12px system-ui"; ctx.textAlign = "center";
      ctx.fillText("Enter valid box lengths and angles to preview the cell.", width / 2, height / 2);
    };
    if (!lengths.every(a => Number.isFinite(a) && a > 0) || !angles.every(a => Number.isFinite(a) && a > 0 && a < 180)) { invalid(); return; }
    const max = Math.max(...lengths), a = lengths[0] / max, b = (lengths[1] || 0) / max, c = (lengths[2] || 0) / max;
    const [alpha, beta, gamma] = angles.map(x => x * Math.PI / 180);
    const cy = c * (Math.cos(alpha) - Math.cos(beta) * Math.cos(gamma)) / Math.sin(gamma);
    const cz2 = c * c - (c * Math.cos(beta)) ** 2 - cy * cy;
    if (dim === 3 && cz2 <= 0) { invalid(); return; }
    const vectors = [[a, 0, 0], [b * Math.cos(gamma), b * Math.sin(gamma), 0], [c * Math.cos(beta), cy, Math.sqrt(Math.max(0, cz2))]];
    const world = coords => [0, 1, 2].map(axis => coords.reduce((sum, f, i) => sum + f * vectors[i][axis], 0));
    const center = world([.5, dim > 1 ? .5 : 0, dim > 2 ? .5 : 0]);
    const rotate = coords => {
      const point = world(coords).map((p, i) => p - center[i]);
      const x = Math.cos(yaw) * point[0] + Math.sin(yaw) * point[2];
      const z = -Math.sin(yaw) * point[0] + Math.cos(yaw) * point[2];
      return [x, Math.cos(pitch) * point[1] - Math.sin(pitch) * z, Math.sin(pitch) * point[1] + Math.cos(pitch) * z];
    };
    const vertices = Array.from({length: 2 ** dim}, (_, i) => rotate([i & 1, (i >> 1) & 1, (i >> 2) & 1]));
    const span = axis => Math.max(...vertices.map(p => p[axis])) - Math.min(...vertices.map(p => p[axis]));
    const scale = Math.min((width - 90) / Math.max(.1, span(0)), (height - 75) / Math.max(.1, span(1)));
    const project = p => [width / 2 + p[0] * scale, height / 2 - p[1] * scale];
    const line = (p, q, stroke, lineWidth = 1) => {
      ctx.beginPath(); ctx.moveTo(...project(p)); ctx.lineTo(...project(q)); ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke();
    };
    if (dim === 3) {
      const faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]];
      faces.sort((a, b) => a.reduce((s, i) => s + vertices[i][2], 0) - b.reduce((s, i) => s + vertices[i][2], 0));
      faces.forEach(face => {
        ctx.beginPath(); face.forEach((i, j) => { const p = project(vertices[i]); if (j) ctx.lineTo(...p); else ctx.moveTo(...p); });
        ctx.closePath(); ctx.fillStyle = "rgba(107,157,139,.055)"; ctx.fill();
      });
    }
    vertices.forEach((point, i) => {
      for (let axis = 0; axis < dim; axis++) if (!(i & 2 ** axis)) line(point, vertices[i | 2 ** axis], "#8ba79b", 1.2);
    });
    model.polymers.slice(0, 12).forEach((polymer, pi) => {
      if (polymer.architecture === "branched") {
        const layout = treeLayout(polymer);
        if (layout.error) return;
        const points = new Map([...layout.positions].map(([id, p]) => [id,
          [.14 + .72 * p[0], dim > 1 ? .14 + .72 * p[1] : 0,
            dim > 2 ? .5 + .18 * Math.sin(p[0] * 4 + p[1] * 3 + pi * 1.1) : 0]]));
        polymer.blocks.forEach(block => {
          const start = points.get(block.v), end = points.get(block.u);
          line(rotate(start), rotate(end), color(block.monomer), 3.5);
        });
        layout.adjacency.forEach((neighbors, id) => {
          if (neighbors.length < 3) return;
          const p = project(rotate(points.get(id)));
          ctx.beginPath(); ctx.arc(...p, 3.5, 0, 2 * Math.PI); ctx.fillStyle = "#233632"; ctx.fill();
        });
        return;
      }
      const lengths = polymer.blocks.map(b => numeric(b.length));
      if (!lengths.every(n => Number.isFinite(n) && n > 0)) return;
      const total = lengths.reduce((a, b) => a + b, 0);
      let start = 0;
      const chain = t => rotate([.12 + .76 * t,
        dim > 1 ? .5 + .21 * Math.sin(t * Math.PI * 2 + pi * 1.1) : 0,
        dim > 2 ? .5 + .21 * Math.cos(t * Math.PI * 2 + pi * 1.1) : 0]);
      polymer.blocks.forEach((block, bi) => {
        const end = start + lengths[bi] / total, steps = Math.max(3, Math.ceil(80 * (end - start)));
        for (let j = 0; j < steps; j++) line(chain(start + (end - start) * j / steps), chain(start + (end - start) * (j + 1) / steps), color(block.monomer), 3.5);
        start = end;
      });
    });
    ctx.fillStyle = "#405d50"; ctx.font = "11px system-ui"; ctx.textAlign = "center";
    for (let axis = 0; axis < dim; axis++) {
      const p = project(vertices[0].map((v, i) => (v + vertices[2 ** axis][i]) / 2));
      ctx.fillText(`${"abc"[axis]} = ${lengths[axis].toPrecision(3)}`, p[0], p[1] + 18);
    }
  }

  function init(molecules, box, onChange) {
    moleculesPanel = molecules; boxPanel = box; notify = onChange;
    const canvas = document.getElementById("box-canvas");
    canvas.addEventListener("pointerdown", event => { drag = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); });
    canvas.addEventListener("pointermove", event => {
      if (!drag) return;
      yaw += (event.clientX - drag[0]) * .008; pitch = Math.max(-1.4, Math.min(1.4, pitch + (event.clientY - drag[1]) * .008));
      drag = [event.clientX, event.clientY]; draw();
    });
    ["pointerup", "pointercancel"].forEach(name => canvas.addEventListener(name, () => { drag = null; }));
    canvas.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      yaw += event.key === "ArrowRight" ? .1 : event.key === "ArrowLeft" ? -.1 : 0;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + (event.key === "ArrowUp" ? -.1 : event.key === "ArrowDown" ? .1 : 0)));
      draw();
    });
    document.getElementById("reset-view").addEventListener("click", () => { yaw = .65; pitch = -.45; draw(); });
    new ResizeObserver(draw).observe(canvas);
  }

  function loadPreset(values) {
    const nx = values.nx.split(/[,\s]+/).filter(Boolean), lx = values.lx.split(/[,\s]+/).filter(Boolean);
    const monomers = [{id: uid(), name: "A", segment_length: values.b_a}, {id: uid(), name: "B", segment_length: values.b_b}];
    model = {box: {dimension: nx.length, shape: {1: "line", 2: "square", 3: "cubic"}[nx.length],
      nx: [...nx, nx[0], nx[0]].slice(0, 3), lx: [...lx, lx[0], lx[0]].slice(0, 3), angles: ["90", "90", "90"]},
      monomers, polymers: [{id: uid(), architecture: "linear", volume_fraction: "1.0", blocks: values.blocks.split(",").map(entry => {
        const [name, length] = entry.split(":").map(s => s.trim());
        return {id: uid(), monomer: monomers.find(m => m.name === name).id, length};
      })}], interactions: [{id: uid(), a: monomers[0].id, b: monomers[1].id, chi_n: values.chi_n}]};
    uniform = nx.every(n => n === nx[0]);
    mode = values.mode;
    renderBox(); renderMolecules(); refresh();
  }
  function serialize() {
    return {box: {...model.box, nx: model.box.nx.slice(0, model.box.dimension), lx: model.box.lx.slice(0, model.box.dimension)},
      monomers: model.monomers.map(m => ({name: m.name, segment_length: m.segment_length})),
      polymers: model.polymers.map(p => ({volume_fraction: p.volume_fraction, blocks: p.blocks.map(b => ({type: byId(b.monomer)?.name || "", length: b.length,
        ...(p.architecture === "branched" ? {v: b.v, u: b.u} : {})}))})),
      interactions: model.interactions.map(p => ({a: byId(p.a)?.name || "", b: byId(p.b)?.name || "", chi_n: p.chi_n}))};
  }
  return {init, loadPreset, setMode, serialize, refresh, dimension: () => model.box.dimension};
})();
