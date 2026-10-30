"use strict";

// The server's API needs the token of the URL that the server printed
const apiHeaders = {"X-Editor-Token": new URLSearchParams(window.location.search).get("token") || ""};
// The home link reloads the page with that token
document.querySelector(".brand").search = window.location.search;

const sections = [
  {id: "system", title: "System", help: "Choose the simulation and calculation method. Edit geometry in Box and chain architectures in Molecules.", fields: [
    ["mode", "Simulation", ["SCFT", "L-FTS", "CL-FTS"], "Load a starting template to apply its parameter defaults. Switching to L-FTS or CL-FTS sets the box to 3D with 90° angles; other settings are kept."],
    ["platform", "Platform", ["auto", "cuda", "cpu-mkl", "cpu-fftw"]],
    ["ds", "Contour step · ds", null, "Fractions such as 1/100 are accepted."],
    ["chain_model", "Chain model", ["continuous", "discrete"]],
    ["numerical_method", "Numerical method", ["auto", "rqm4", "rk2", "cn-adi2"], "CN-ADI2 supports real fields in orthogonal cells; it is unavailable for CL-FTS."],
    ["threads", "OpenMP threads", null, "1–4"],
    ["reduce_memory", "Reduce memory usage", "checkbox", "Enable only when memory is insufficient. This reduces memory usage but slows down the simulation."],
    ["space_group", "Space group (SCFT only)", [], "Choose None to use the full grid. Hall numbers distinguish settings of the same space group.", true]
  ]},
  {id: "box", title: "Box", help: "", fields: []},
  {id: "molecules", title: "Molecules", help: "", fields: []},
  {id: "scft", title: "SCFT", help: "Set the stopping criteria for the self-consistent field calculation.", fields: [
    ["max_iter", "Maximum iterations"],
    ["tolerance", "Convergence tolerance"],
    ["relax_cell", "Relax cell lengths and allowed angles", "checkbox", "Uses the selected cell shape, or the space group's crystal system. 2D Square cells require a fixed cell."]
  ]},
  {id: "lfts", title: "Dynamics", help: "Set Langevin dynamics and recording intervals for L-FTS or CL-FTS. Results are saved under data/ in the output directory.", fields: [
    ["max_step", "Final Langevin step · max_step", null, "Final step number, including resumed steps. Loading initial fields starts a new run."], ["dt", "Langevin time step · dt"],
    ["nbar", "Invariant polymerization · nbar"], ["compressor", "Field compressor", ["am", "lr", "lram"]],
    ["saddle_max_iter", "Saddle maximum iterations"], ["saddle_tolerance", "Saddle tolerance"],
    ["checkpoint_interval", "Checkpoint interval (steps)", null, "Save the full state for restarting the simulation."],
    ["sample_interval", "Observable sampling interval (steps)", null, "Collect samples of observables, including energy and structure functions."],
    ["statistics_save_interval", "Statistics save interval (steps)", null, "Save accumulated observables and reset the accumulators. A multiple of the sampling interval."]
  ]},
  {id: "clfts", title: "CL-FTS", help: "All fields fluctuate in complex Langevin dynamics. Set compressibility, dynamical stabilization and optional finite-range interactions.", fields: [
    ["zeta_n", "Compressibility · zeta_n", null, "Finite, non-negative value; default 100."],
    ["dynamic_stabilization", "Dynamical stabilization · alpha_ds", null, "Non-negative; 0 disables stabilization. The appropriate value depends on your system."],
    ["smearing", "Interaction smearing", ["none", "gaussian", "sigmoidal"]],
    ["a_int", "Gaussian length · a_int", null, "In reference-chain length units; 0 disables smearing."],
    ["k_int", "Sigmoidal cutoff · k_int"], ["dk_int", "Sigmoidal width · dk_int"]
  ]},
  {id: "mixing", title: "Mixing", help: "Anderson mixing settings for SCFT or L-FTS field compression.", fields: [
    ["max_hist", "History length · max_hist"], ["start_error", "AM start error · start_error"],
    ["mix_min", "Minimum mixing rate · mix_min"], ["mix_init", "Initial mixing rate · mix_init"]
  ]},
  {id: "fields", title: "Fields / Output", help: "This beta release includes lamella and gyroid phase-specific initial fields. Use random or mat fields for other phases or more than two monomer types. For L-FTS or CL-FTS, load converged SCFT fields or allow sufficient equilibration. CL-FTS also accepts complex fields from mat files.", fields: [
    ["initial", "Initial fields", ["lamella", "gyroid", "random", "mat"]],
    ["amplitude", "Seed scale / random std"],
    ["seed", "Random seed", null, "Leave blank to choose a new seed for each run.", true],
    ["mat_file", "Initial-field .mat path", null, "File containing w_<name> for every monomer. CL-FTS also accepts w_<name>_real and w_<name>_imag pairs. Enter a path on the simulation machine; the editor does not upload files.", true],
    ["output_dir", "Simulation output directory", null, "Relative to the saved Python script. Use a separate directory for each run.", true]
  ]}
];

const form = document.getElementById("settings-form");
const tabs = document.getElementById("tabs");
const panels = document.getElementById("panels");
const presetSelect = document.getElementById("preset");
const code = document.getElementById("script");
const feedback = document.getElementById("feedback");
const copyButton = document.getElementById("copy");
const downloadButton = document.getElementById("download");
const serverSaveButton = document.getElementById("server-save");
const inputs = new Map();
const tabButtons = new Map();
let presets;
let spaceGroups;
let activeTab = "system";
let revision = 0;
let requestId = 0;
let renderedRevision = -1;
let lastScript = "";
let timer;
let controller;

function status(message, state = "loading") {
  feedback.textContent = message;
  feedback.dataset.state = state;
}

function activateTab(id) {
  activeTab = id;
  for (const [name, button] of tabButtons) {
    const selected = name === id;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    document.getElementById(`panel-${name}`).hidden = !selected;
  }
}

function renderForm() {
  for (const section of sections) {
    const button = document.createElement("button");
    button.type = "button";
    button.id = `tab-${section.id}`;
    button.textContent = section.title;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", `panel-${section.id}`);
    button.addEventListener("click", () => activateTab(section.id));
    tabs.append(button);
    tabButtons.set(section.id, button);

    const panel = document.createElement("div");
    panel.id = `panel-${section.id}`;
    panel.className = "tab-panel";
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", button.id);
    const help = document.createElement("p");
    help.className = "section-help";
    help.textContent = section.help;
    panel.append(help);
    const grid = document.createElement("div");
    grid.className = "field-grid";
    for (const [key, title, options, hint, full] of section.fields) {
      const row = document.createElement("div");
      row.className = `field${full || options === "checkbox" ? " full" : ""}${options === "checkbox" ? " checkbox" : ""}`;
      const label = document.createElement("label");
      label.htmlFor = key;
      label.textContent = title;
      const input = document.createElement(Array.isArray(options) ? "select" : "input");
      input.id = key;
      input.name = key;
      if (Array.isArray(options)) {
        for (const option of options) {
          const [value, title] = Array.isArray(option) ? option : [option, option];
          input.add(new Option(title, value));
        }
      } else {
        input.type = options === "checkbox" ? "checkbox" : "text";
        input.spellcheck = false;
        input.maxLength = 512;
      }
      if (options === "checkbox") {
        label.prepend(input);
        row.append(label);
      } else {
        row.append(label, input);
      }
      if (hint) {
        const note = document.createElement("span");
        note.id = `${key}-hint`;
        note.className = "field-hint";
        note.textContent = hint;
        input.setAttribute("aria-describedby", note.id);
        row.append(note);
      }
      inputs.set(key, input);
      grid.append(row);
    }
    panel.append(grid);
    panels.append(panel);
  }
  ModelEditor.init(document.getElementById("panel-molecules"), document.getElementById("panel-box"), changed);
  document.getElementById("panel-molecules").dataset.modelEditor = "true";
  document.getElementById("panel-box").dataset.modelEditor = "true";
  activateTab(activeTab);
}

function settings() {
  return {...Object.fromEntries([...inputs].map(([key, input]) => [key, input.type === "checkbox" ? input.checked : input.value])),
    model: ModelEditor.serialize()};
}

function updateConditionalFields() {
  const mode = inputs.get("mode").value;
  ModelEditor.setMode(mode);
  inputs.get("space_group").disabled = mode !== "SCFT";
  const group = spaceGroups.get(inputs.get("space_group").value);
  const groupHint = document.getElementById("space_group-hint");
  if (mode !== "SCFT") {
    groupHint.textContent = "Space group symmetry is available only for SCFT. The selection is remembered for SCFT and excluded from FTS scripts.";
  } else if (!group) {
    groupHint.textContent = "Choose None to use the full grid. Hall numbers distinguish settings of the same space group.";
  } else {
    const cell = group.crystal_system === "Cubic" ? "a = b = c; angles 90°" :
      group.crystal_system === "Tetragonal" ? "a = b; angles 90° (choose Tetragonal)" :
      group.crystal_system === "Orthorhombic" ? "angles 90°" :
      group.crystal_system === "Monoclinic" ? `${["α", "β", "γ"].filter((_, i) => i !== group.unique_axis).join(" = ")} = 90°, ` +
        `${["α", "β", "γ"][group.unique_axis]} free (unique axis ${"abc"[group.unique_axis]}; use ${group.unique_axis === 1 ? "Monoclinic" : "Triclinic"})` :
      group.crystal_system === "Triclinic" ? "any cell (use Triclinic)" :
      group.hexagonal_axes ? "a = b; angles 90°, 90°, 120° (choose Hexagonal)" :
      "a = b = c; equal angles; disable box relaxation (use Triclinic)";
    const grid = [...group.equal_grid_axes.map(([a, b]) => `n${"xyz"[a]} = n${"xyz"[b]}`),
      ...group.grid_divisors.flatMap((divisor, axis) => divisor > 1 ? [`n${"xyz"[axis]} a multiple of ${divisor}`] : [])];
    groupHint.textContent = `${group.crystal_system} · Hall ${group.hall}. Requires 3D; ${cell}.${grid.length ? ` Grid: ${grid.join(", ")}.` : ""} Symmetry does not select an initial phase. Use compatible initial fields, such as random or mat; fields are projected onto the selected symmetry.`;
  }
  tabButtons.get("scft").hidden = mode !== "SCFT";
  tabButtons.get("lfts").hidden = mode === "SCFT";
  tabButtons.get("clfts").hidden = mode !== "CL-FTS";
  tabButtons.get("mixing").hidden = mode === "CL-FTS";
  for (const key of ["compressor", "saddle_max_iter", "saddle_tolerance"]) {
    inputs.get(key).closest(".field").hidden = mode !== "L-FTS";
  }
  const smearing = inputs.get("smearing").value;
  inputs.get("a_int").closest(".field").hidden = smearing !== "gaussian";
  for (const key of ["k_int", "dk_int"]) inputs.get(key).closest(".field").hidden = smearing !== "sigmoidal";
  if (tabButtons.get(activeTab).hidden) activateTab(mode === "SCFT" ? "scft" : "lfts");
  const discrete = inputs.get("chain_model").value === "discrete";
  const method = inputs.get("numerical_method");
  method.disabled = discrete;
  for (const option of method.options) option.disabled = mode === "CL-FTS" && option.value === "cn-adi2";
  if (discrete || mode === "CL-FTS" && method.value === "cn-adi2") method.value = "auto";
  const box = ModelEditor.serialize().box;
  const fixedSquare = mode === "SCFT" && box.dimension === 2 && box.shape === "square";
  inputs.get("relax_cell").disabled = fixedSquare;
  if (fixedSquare) inputs.get("relax_cell").checked = false;
  const mat = inputs.get("initial").value === "mat";
  inputs.get("mat_file").closest(".field").hidden = !mat;
  inputs.get("amplitude").disabled = mat;
  const dimensions = ModelEditor.dimension();
  document.getElementById("system-summary").textContent = `${mode} · ${dimensions}D`;
  ModelEditor.refresh();
}

function changed() {
  revision += 1;
  copyButton.disabled = true;
  updateConditionalFields();
  status("Updating the preview with your changes.");
  clearTimeout(timer);
  timer = setTimeout(generate, 350);
}

async function generate() {
  clearTimeout(timer);
  if (controller) controller.abort();
  controller = new AbortController();
  const thisRequest = ++requestId;
  const thisRevision = revision;
  document.getElementById("code-preview").setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {"Content-Type": "application/json", ...apiHeaders},
      body: JSON.stringify(settings()),
      signal: controller.signal
    });
    const result = await response.json();
    if (thisRequest !== requestId || thisRevision !== revision) return null;
    if (!response.ok) throw new Error(result.error || "Could not generate the script.");
    code.textContent = result.script;
    lastScript = result.script;
    renderedRevision = thisRevision;
    copyButton.disabled = false;
    status("Script ready. Save on the server or download to your computer.", "ready");
    return lastScript;
  } catch (error) {
    if (error.name === "AbortError" || thisRequest !== requestId || thisRevision !== revision) return null;
    copyButton.disabled = true;
    status(error instanceof TypeError ? "Check the server connection. Keep the editor and SSH tunnel running." : error.message, "error");
    return null;
  } finally {
    if (thisRequest === requestId) document.getElementById("code-preview").setAttribute("aria-busy", "false");
  }
}

function loadPreset() {
  const values = presets[presetSelect.value];
  for (const [key, value] of Object.entries(values)) {
    const input = inputs.get(key);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = value;
    else input.value = value;
  }
  ModelEditor.loadPreset(values);
  revision += 1;
  copyButton.disabled = true;
  updateConditionalFields();
  status("Generating a preview from the template.");
  generate();
}

function exportFilename() {
  let name = document.getElementById("filename").value.trim().replace(/[\\/:]/g, "_") || "simulation.py";
  if (!name.toLowerCase().endsWith(".py")) name += ".py";
  return name;
}

serverSaveButton.addEventListener("click", async () => {
  serverSaveButton.disabled = true;
  clearTimeout(timer);
  if (controller) controller.abort();
  requestId += 1;
  const thisRevision = revision;
  const resultLabel = document.getElementById("server-save-result");
  resultLabel.hidden = false;
  resultLabel.dataset.state = "loading";
  resultLabel.textContent = "Saving the script on the server.";
  try {
    // Keep this write request independent of cancellable preview requests.
    const response = await fetch("/api/save", {
      method: "POST",
      headers: {"Content-Type": "application/json", ...apiHeaders},
      body: JSON.stringify({
        values: settings(),
        filename: exportFilename(),
        directory: document.getElementById("server-subdir").value.trim()
      })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(response.status === 409 ?
      "This file already exists. Choose another filename or server subfolder." : result.error);
    resultLabel.dataset.state = "ready";
    resultLabel.textContent = `Saved on server: ${result.path}`;
    if (thisRevision === revision) {
      code.textContent = result.script;
      lastScript = result.script;
      renderedRevision = thisRevision;
      copyButton.disabled = false;
      status("Current settings saved on the server.", "ready");
    } else {
      status("Saved the settings from when you clicked Save. Save again to include later edits.");
    }
  } catch (error) {
    resultLabel.dataset.state = "error";
    resultLabel.textContent = error instanceof TypeError ?
      "Check the server connection. If no response arrived, check the server folder to see whether the file was saved." : error.message;
    status("Could not complete the server save. Check the save message above.", "error");
  } finally {
    serverSaveButton.disabled = false;
    document.getElementById("code-preview").setAttribute("aria-busy", "false");
  }
});

downloadButton.addEventListener("click", async () => {
  downloadButton.disabled = true;
  try {
    const script = await generate();
    if (!script) return;
    const name = exportFilename();
    const url = URL.createObjectURL(new Blob([script], {type: "text/x-python;charset=utf-8"}));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status(`Download requested: ${name}. Check your browser's download folder.`, "ready");
  } finally {
    downloadButton.disabled = false;
  }
});

copyButton.addEventListener("click", async () => {
  if (renderedRevision !== revision) return;
  try {
    await navigator.clipboard.writeText(lastScript);
    status("Current script copied to the clipboard.", "ready");
  } catch (_) {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    status("Code selected. Press Ctrl+C or Cmd+C to copy.");
  }
});

tabs.addEventListener("keydown", event => {
  const visible = [...tabButtons.values()].filter(button => !button.hidden);
  const index = visible.indexOf(document.activeElement);
  if (index < 0 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const next = event.key === "Home" ? 0 : event.key === "End" ? visible.length - 1 :
    (index + (event.key === "ArrowRight" ? 1 : -1) + visible.length) % visible.length;
  visible[next].click();
  visible[next].focus();
});
const formChanged = event => { if (!event.target.closest("[data-model-editor]")) changed(); };
form.addEventListener("input", formChanged);
form.addEventListener("change", formChanged);
form.addEventListener("submit", event => { event.preventDefault(); generate(); });
document.getElementById("load-preset").addEventListener("click", loadPreset);
document.getElementById("refresh").addEventListener("click", generate);

async function start() {
  try {
    const [response, configResponse] = await Promise.all([fetch("/api/presets", {headers: apiHeaders}), fetch("/api/config", {headers: apiHeaders})]);
    const refused = [response, configResponse].find(r => r.status === 403);
    if (refused) throw new Error((await refused.json().catch(() => ({}))).error || "The server refused the request.");
    if (!response.ok || !configResponse.ok) throw new Error("Could not load the editor settings. Check the server and refresh the page.");
    const config = await configResponse.json();
    document.getElementById("server-root").textContent = config.save_dir;
    spaceGroups = new Map(config.space_groups.map(group => [String(group.hall), group]));
    sections.find(section => section.id === "system").fields.find(field => field[0] === "space_group")[2] =
      [["none", "None (full grid)"], ...config.space_groups.map(group => [String(group.hall), group.label])];
    presets = await response.json();
    renderForm();
    presetSelect.replaceChildren(...Object.keys(presets).map(name => new Option(name, name)));
    for (const id of ["preset", "load-preset", "refresh", "download", "server-save"]) document.getElementById(id).disabled = false;
    loadPreset();
  } catch (error) {
    status(`Could not open the editor. ${error.message}`, "error");
  }
}
start();
