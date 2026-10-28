#!/usr/bin/env python3
"""Simulation Script Builder (beta): export SCFT/L-FTS/CL-FTS scripts without running simulations.

Run on a headless Linux server and connect through an SSH tunnel. Only the
Python standard library is needed by the editor; no Tk, GPU or polymerfts install.
Phase-specific initial fields currently support lamella and gyroid.
"""

import argparse
import hmac
import json
import math
import os
import re
import secrets
from fractions import Fraction
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pprint import pformat
from textwrap import dedent, indent
from urllib.parse import urlsplit


PRESETS = ("SCFT: AB lamella", "SCFT: AB gyroid", "SCFT: ABA lamella",
           "L-FTS: AB lamella", "CL-FTS: AB lamella")


@lru_cache(maxsize=1)
def space_groups():
    """Offline Hall settings; neither spglib nor polymerfts is needed by the editor."""
    path = Path(__file__).parent / "assets" / "space-groups.json"
    return {str(group["hall"]): group for group in json.loads(path.read_text(encoding="utf-8"))["settings"]}


def space_group_options():
    common = {529: "BCC / DP", 530: "Gyroid", 523: "FCC", 517: "SC / SP", 520: "A15",
              526: "Diamond", 522: "DD", 510: "SG", 419: "Sigma", 488: "HCP / PL", 336: "Fddd"}
    ordered = sorted(space_groups().values(), key=lambda g: (g["hall"] not in common, g["symbol"], g["hall"]))
    return [{**group, "label": f"{group['symbol']} · Hall {group['hall']}"
             + (f" · {common[group['hall']]}" if group["hall"] in common else "")
             + (f" · setting {group['setting']}" if group["setting"] else "")} for group in ordered]


def apply_space_group(params, selection, relax_cell):
    """Validate the selected SCFT cell/grid before adding symmetry to its script."""
    group = space_groups().get(selection)
    if group is None:
        raise ValueError("Choose a supported space group Hall setting, or None.")
    if len(params["nx"]) != 3:
        raise ValueError("Space group symmetry requires a 3D box. Choose 3D in Box.")
    system = group["crystal_system"]
    lx, angles = params["lx"], params.get("angles", [90, 90, 90])
    same = lambda a, b: math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    orthogonal = all(abs(a - 90) < 1e-8 for a in angles)
    if system == "Cubic":
        valid = same(lx[0], lx[1]) and same(lx[0], lx[2]) and orthogonal
        need = "a = b = c and angles [90, 90, 90]"
    elif system == "Tetragonal":
        valid = same(lx[0], lx[1]) and orthogonal
        need = "a = b and angles [90, 90, 90] (use an Orthorhombic box with a = b)"
    elif system == "Orthorhombic":
        valid, need = orthogonal, "angles [90, 90, 90]"
    elif group["hexagonal_axes"]:
        valid = same(lx[0], lx[1]) and abs(angles[0] - 90) < 1e-8 and abs(angles[1] - 90) < 1e-8 and same(angles[2], 120)
        need = "a = b and angles [90, 90, 120] (choose Hexagonal in Box)"
    else:
        valid = same(lx[0], lx[1]) and same(lx[0], lx[2]) and same(angles[0], angles[1]) and same(angles[0], angles[2])
        need = "a = b = c and alpha = beta = gamma (use a Triclinic box)"
        if valid and relax_cell:
            raise ValueError("Rhombohedral-axis space groups require fixed box size. Disable box relaxation or choose a hexagonal-axis Hall setting.")
    if not valid:
        raise ValueError(f"Space group {group['symbol']} (Hall {group['hall']}) requires {need}.")
    nx = params["nx"]
    for a, b in group["equal_grid_axes"]:
        if nx[a] != nx[b]:
            raise ValueError(f"Space group {group['symbol']} requires n{'xyz'[a]} = n{'xyz'[b]}. Edit the grid in Box.")
    for axis, divisor in enumerate(group["grid_divisors"]):
        if nx[axis] % divisor:
            raise ValueError(f"Space group {group['symbol']} requires n{'xyz'[axis]} to be a multiple of {divisor}.")
    params["space_group"] = {"symbol": group["symbol"], "hall_number": group["hall"]}


def preset_settings(name):
    """Return editable form values; lengths are relative to the reference chain."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    values = dict(
        mode="SCFT", nx="64, 64, 64", lx="4.0, 4.0, 4.0", ds="1/100", chain_model="continuous",
        platform="auto", numerical_method="auto", threads="2", reduce_memory=False,
        space_group="none",
        blocks="A:0.5, B:0.5", b_a="1.0", b_b="1.0", chi_n="20.0",
        max_iter="5000", tolerance="1e-8", relax_cell=True,
        max_hist="20", start_error="1e-2", mix_min="0.1", mix_init="0.1",
        max_step="500000", dt="8.0", nbar="10000", compressor="am",
        saddle_max_iter="100", saddle_tolerance="1e-4",
        zeta_n="100.0", dynamic_stabilization="0.0", smearing="none",
        a_int="0.02", k_int="20.0", dk_int="5.0",
        checkpoint_interval="1000", sample_interval="10", statistics_save_interval="10000",
        initial="lamella", amplitude="1.0", seed="12345", mat_file="",
        output_dir="simulation",
    )
    if name == "SCFT: AB gyroid":
        values.update(nx="32, 32, 32", lx="3.3, 3.3, 3.3", blocks="A:0.36, B:0.64",
                      initial="gyroid")
    elif name == "SCFT: ABA lamella":
        values.update(nx="256", lx="1.38", ds="1/200", chi_n="9.5",
                      blocks="A:0.5, B:1.0, A:0.5", max_iter="2000")
    elif name == "L-FTS: AB lamella":
        values.update(mode="L-FTS", nx="40, 40, 40", lx="4.36, 4.36, 4.36",
                      ds="1/90", chain_model="discrete", blocks="A:0.4, B:0.6",
                      chi_n="17.0", mix_min="0.01", mix_init="0.01", start_error="0.5")
    elif name == "CL-FTS: AB lamella":
        # Parameter defaults from examples/block_copolymer/clfts/Lamella.py.
        values.update(mode="CL-FTS", nx="32, 32, 32", lx="4.21, 4.21, 4.21",
                      ds="1/90", chain_model="discrete", chi_n="12.0",
                      max_step="100000", dt="1.0", amplitude="2.0",
                      dynamic_stabilization="0.01")
    return values


def number(text, label, *, positive=True, integer=False):
    """Accept decimals, scientific notation and fractions, without eval()."""
    try:
        text = str(text).strip()
        exponent = re.search(r"[eE]([+-]?\d+)$", text)
        if len(text) > 128 or (exponent and abs(int(exponent[1])) > 1000):
            raise ValueError("Number is too large")
        fraction = Fraction(text)
        value = int(fraction) if integer and fraction.denominator == 1 else float(fraction)
    except (ValueError, ZeroDivisionError, OverflowError):
        raise ValueError(f"{label}: enter a finite number (e.g. 0.01 or 1/100).") from None
    if (not integer and not math.isfinite(value)) or (positive and value <= 0):
        raise ValueError(f"{label}: enter a finite {'positive ' if positive else ''}number.")
    if integer:
        if fraction.denominator != 1:
            raise ValueError(f"{label}: enter an integer.")
        return int(value)
    return value


def validate_polymer_graph(blocks, label):
    """Check explicit block endpoints against the solver's connected-tree contract."""
    parents, edges = {}, set()

    def root(vertex):
        while parents[vertex] != vertex:
            parents[vertex] = parents[parents[vertex]]
            vertex = parents[vertex]
        return vertex

    for block in blocks:
        v, u = block["v"], block["u"]
        if v == u:
            raise ValueError(f"{label}: a block must connect two different nodes.")
        edge = tuple(sorted((v, u)))
        if edge in edges:
            raise ValueError(f"{label}: duplicate connection between nodes {edge[0]} and {edge[1]}.")
        edges.add(edge)
        parents.setdefault(v, v)
        parents.setdefault(u, u)
        rv, ru = root(v), root(u)
        if rv == ru:
            raise ValueError(f"{label}: a cycle was detected. Ring polymers are not supported; use a tree.")
        parents[ru] = rv
    if len({root(vertex) for vertex in parents}) != 1:
        raise ValueError(f"{label}: all blocks must belong to one connected polymer.")


def model_parameters(model):
    """Validate the browser's editable rows and simulation cell."""
    def row(value, keys, label):
        if not isinstance(value, dict) or value.keys() != set(keys.split()):
            raise ValueError(f"{label}: missing or unexpected fields.")

    row(model, "box monomers polymers interactions", "Model")
    box = model["box"]
    row(box, "dimension shape nx lx angles", "Box")
    dim = box["dimension"]
    shapes = {1: ("line",), 2: ("square", "rectangle"),
              3: ("cubic", "orthorhombic", "hexagonal", "triclinic")}
    if type(dim) is not int or dim not in shapes or box["shape"] not in shapes[dim]:
        raise ValueError("Choose a box shape matching the dimension.")
    for key in ("nx", "lx"):
        if not isinstance(box[key], list) or len(box[key]) != dim:
            raise ValueError(f"Box {key}: enter one value per dimension.")
    nx = [number(n, "Grid points", integer=True) for n in box["nx"]]
    lx = [number(n, "Box length") for n in box["lx"]]
    if not isinstance(box["angles"], list) or len(box["angles"]) != 3:
        raise ValueError("Box angles must contain alpha, beta and gamma.")
    angles = [number(a, "Box angle") for a in box["angles"]]
    if any(a >= 180 for a in angles):
        raise ValueError("Box angles must be between 0 and 180 degrees.")
    shape = box["shape"]
    equal = lx if shape in ("square", "cubic") else lx[:2] if shape == "hexagonal" else []
    if equal and not all(math.isclose(a, equal[0], rel_tol=1e-10) for a in equal):
        raise ValueError(f"{shape.title()} boxes require equal {'first two ' if shape == 'hexagonal' else ''}lengths.")
    expected = [90, 90, 120] if shape == "hexagonal" else [90, 90, 90]
    if shape != "triclinic" and angles != expected:
        raise ValueError(f"{shape.title()} box angles must be {expected} degrees.")
    ca, cb, cg = [math.cos(math.radians(a)) for a in angles]
    # The solver rejects cells flatter than V = 0.1 abc; the small margin keeps
    # rounding from letting through a cell that it rejects
    if 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg < 0.1 * 0.1 * (1 + 1e-9):
        raise ValueError("Box angles must form a cell of volume at least 0.1 abc; "
                         "use the reduced (Niggli) cell of the lattice, spanned by its shortest vectors, "
                         "or for a rhombohedral-axis Hall setting the hexagonal-axis setting of the group.")

    monomers = model["monomers"]
    if not isinstance(monomers, list) or not 2 <= len(monomers) <= 32:
        raise ValueError("Add between 2 and 32 monomer types.")
    segments = {}
    for monomer in monomers:
        row(monomer, "name segment_length", "Monomer")
        name = monomer["name"]
        # The library takes letters and underscores only (Polymer.cpp)
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z_]{0,15}", name):
            raise ValueError("Monomer names must start with a letter and contain up to 16 letters or underscores.")
        if name in segments:
            raise ValueError(f"Duplicate monomer name: {name}.")
        segments[name] = number(monomer["segment_length"], f"Segment length {name}")
    polymers = model["polymers"]
    if not isinstance(polymers, list) or not 1 <= len(polymers) <= 64:
        raise ValueError("Add between 1 and 64 polymers.")
    distinct = []
    used = set()
    for i, polymer in enumerate(polymers, 1):
        row(polymer, "volume_fraction blocks", f"Polymer {i}")
        vf = number(polymer["volume_fraction"], f"Polymer {i} volume fraction", positive=False)
        if not 0 <= vf <= 1:
            raise ValueError(f"Polymer {i}: volume fraction must be between 0 and 1.")
        if not isinstance(polymer["blocks"], list) or not 1 <= len(polymer["blocks"]) <= 256:
            raise ValueError(f"Polymer {i}: add between 1 and 256 blocks.")
        blocks = []
        connected = isinstance(polymer["blocks"][0], dict) and bool({"v", "u"} & polymer["blocks"][0].keys())
        for block in polymer["blocks"]:
            row(block, "type length v u" if connected else "type length", f"Polymer {i} block")
            if not isinstance(block["type"], str) or block["type"] not in segments:
                raise ValueError(f"Polymer {i}: choose a defined monomer for every block.")
            entry = {"type": block["type"], "length": number(block["length"], "Block length")}
            if connected:
                for key in ("v", "u"):
                    if type(block[key]) not in (str, int):
                        raise ValueError(f"Polymer {i}: node IDs must be non-negative integers.")
                    vertex = number(block[key], f"Polymer {i} node {key}", positive=False, integer=True)
                    if not 0 <= vertex <= 2147483647:
                        raise ValueError(f"Polymer {i}: node IDs must be between 0 and 2147483647.")
                    entry[key] = vertex
            blocks.append(entry)
            used.add(block["type"])
        if connected:
            validate_polymer_graph(blocks, f"Polymer {i}")
        distinct.append({"volume_fraction": vf, "blocks": blocks})
    if not math.isclose(sum(p["volume_fraction"] for p in distinct), 1, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Polymer volume fractions must sum to 1. Use Equal fractions or edit the values.")
    if used != set(segments):
        raise ValueError("Every monomer type must appear in at least one polymer block.")
    interactions = model["interactions"]
    if not isinstance(interactions, list) or len(interactions) > 496:
        raise ValueError("Interactions must be a list of monomer pairs.")
    chi_n = {}
    for interaction in interactions:
        row(interaction, "a b chi_n", "Interaction")
        a, b = interaction["a"], interaction["b"]
        if not isinstance(a, str) or not isinstance(b, str) or a not in segments or b not in segments or a == b:
            raise ValueError("Interactions must connect two different defined monomers.")
        pair = ",".join(sorted((a, b)))
        if pair in chi_n:
            raise ValueError(f"Duplicate interaction: {pair}.")
        chi_n[pair] = number(interaction["chi_n"], f"Interaction {pair}", positive=False)
    names = list(segments)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            chi_n.setdefault(",".join(sorted((a, b))), 0.0)
    params = dict(nx=nx, lx=lx, segment_lengths=segments, chi_n=chi_n, polymers=distinct)
    if angles != [90, 90, 90]:
        params["angles"] = angles
    return params


def imaginary_exchange_fields(types, chi_n):
    """Number of imaginary auxiliary fields besides the pressure field of an incompressible
    model: the positive eigenvalues of chi N on compositions with zero sum. By Sylvester's law
    of inertia the basis e_i - e_last gives the same count as the orthonormal one of
    polymer_field_theory; the eigenvalues come from cyclic Jacobi rotations."""
    chi = lambda a, b: 0.0 if a == b else chi_n.get(f"{a},{b}", chi_n.get(f"{b},{a}", 0.0))
    last, rest = types[-1], types[:-1]
    m = [[chi(a, b) - chi(a, last) - chi(last, b) for b in rest] for a in rest]
    n = len(m)
    for _ in range(100):
        if all(abs(m[p][q]) < 1e-14 * (1.0 + abs(m[p][p]) + abs(m[q][q])) for p in range(n) for q in range(p + 1, n)):
            break
        for p in range(n):
            for q in range(p + 1, n):
                if m[p][q] == 0.0:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                m[p][p] -= t * m[p][q]
                m[q][q] += t * m[p][q]
                m[p][q] = m[q][p] = 0.0
                for r in range(n):
                    if r not in (p, q):
                        rp, rq = m[r][p], m[r][q]
                        m[r][p] = m[p][r] = c * rp - s * rq
                        m[r][q] = m[q][r] = s * rp + c * rq
    return sum(m[i][i] > 1e-8 for i in range(n))


def build_script(values):
    """Validate form values and render a standalone, editable Python script."""
    def num(key, **kwargs):
        return number(values[key], key, **kwargs)

    def choice(key, options):
        if values[key] not in options:
            raise ValueError(f"{key}: choose one of {', '.join(options)}.")
        return values[key]

    mode = choice("mode", ("SCFT", "L-FTS", "CL-FTS"))
    structured = model_parameters(values["model"]) if "model" in values else None
    nx = structured["nx"] if structured else [number(n, "nx", integer=True) for n in values["nx"].replace(",", " ").split()]
    lx = structured["lx"] if structured else [number(n, "lx") for n in values["lx"].replace(",", " ").split()]
    if len(nx) not in (1, 2, 3) or len(lx) != len(nx):
        raise ValueError("nx and lx must have the same number of entries (1, 2 or 3).")
    if mode != "SCFT" and (len(nx) != 3 or structured is not None and "angles" in structured):
        raise ValueError("L-FTS and CL-FTS require a 3D orthogonal (non-skewed) box with all angles equal to 90 degrees.")
    ds = num("ds")
    if ds > 1:
        raise ValueError("ds must be in (0, 1].")
    blocks = []
    for entry in values.get("blocks", "").split(",") if structured is None else []:
        parts = entry.strip().split(":")
        if len(parts) != 2 or parts[0].strip() not in ("A", "B"):
            raise ValueError("Blocks: use A:0.5, B:0.5 (A/B linear chains only).")
        blocks.append({"type": parts[0].strip(), "length": number(parts[1], "block length")})
    if structured is None and {block["type"] for block in blocks} != {"A", "B"}:
        raise ValueError("The chain must contain both A and B blocks.")
    threads = num("threads", integer=True)
    if threads > 4:
        raise ValueError("OpenMP threads must be between 1 and 4.")
    seed = None if not values["seed"].strip() else num("seed", positive=False, integer=True)
    if seed is not None and seed < 0:
        raise ValueError("Random seed must be non-negative, or empty for a random seed.")
    initial = choice("initial", ("lamella", "gyroid", "random", "mat"))
    if initial == "gyroid" and len(nx) != 3:
        raise ValueError("Gyroid initial fields require a 3D grid.")
    mat_file = values["mat_file"].strip()
    if initial == "mat" and not mat_file:
        raise ValueError("Choose a .mat file containing initial fields for every monomer.")
    output_dir = values["output_dir"].strip()
    if not output_dir or "\0" in output_dir or "\0" in mat_file:
        raise ValueError("Enter an output directory; paths cannot contain null characters.")

    params = structured if structured is not None else dict(nx=nx, lx=lx,
                  segment_lengths={"A": num("b_a"), "B": num("b_b")},
                  chi_n={"A,B": num("chi_n", positive=False)},
                  polymers=[{"volume_fraction": 1.0, "blocks": blocks}])
    params.update(chain_model=choice("chain_model", ("continuous", "discrete")), ds=ds,
                  reduce_memory=bool(values["reduce_memory"]))
    if params["chain_model"] == "discrete":
        # The library's rule (Molecules::add_polymer): a whole number of segments per block
        for polymer in params["polymers"]:
            for block in polymer["blocks"]:
                n_segment = round(block["length"] / ds)
                if abs(block["length"] - n_segment * ds) > 1e-9:
                    raise ValueError(f"Discrete chains: block length {block['length']:g} must be a multiple of "
                                     f"ds = {ds:g} (closest: {n_segment * ds:g}).")
    platform = choice("platform", ("auto", "cuda", "cpu-mkl", "cpu-fftw"))
    if platform != "auto":
        params["platform"] = platform
    method = choice("numerical_method", ("auto", "rqm4", "rk2", "cn-adi2"))
    if method != "auto":
        if params["chain_model"] == "discrete":
            raise ValueError("Choose numerical method 'auto' for discrete chains.")
        params["numerical_method"] = method
    if method == "cn-adi2" and "angles" in params:
        raise ValueError("For non-orthogonal boxes choose a pseudo-spectral method.")
    if mode == "CL-FTS" and method == "cn-adi2":
        raise ValueError("CN-ADI2 does not support complex fields. For CL-FTS choose auto, rqm4 or rk2.")
    if mode != "CL-FTS":
        mixing = dict(name="am", max_hist=num("max_hist", integer=True),
                      start_error=num("start_error"), mix_min=num("mix_min"), mix_init=num("mix_init"))
        if max(mixing["mix_min"], mixing["mix_init"]) > 1:
            raise ValueError("Mixing rates must be in (0, 1].")
    if mode == "SCFT":
        params.update(relax_cell=bool(values["relax_cell"]), stress_interval=1,
                      optimizer=mixing, max_iter=num("max_iter", integer=True),
                      tolerance=num("tolerance"))
        if values.get("space_group", "none") != "none":
            apply_space_group(params, values["space_group"], params["relax_cell"])
        elif structured is not None:
            shape = values["model"]["box"]["shape"]
            if shape == "square" and params["relax_cell"]:
                raise ValueError("2D Square cells require fixed-cell SCFT. Disable cell relaxation or choose Rectangle.")
            # A space group determines its own crystal system. Otherwise preserve
            # the cell constraints selected in the editor during SCFT relaxation.
            params["crystal_system"] = (shape.title() if len(nx) == 3 else "Orthorhombic")
            if method == "cn-adi2" and shape == "triclinic" and params["relax_cell"]:
                raise ValueError("CN-ADI2 requires orthogonal cells. Choose a pseudo-spectral method to relax triclinic angles.")
    else:
        params.update(
            langevin=dict(max_step=num("max_step", integer=True), dt=num("dt"), nbar=num("nbar")),
            verbose_level=1,
            recording=dict(dir="data", checkpoint_interval=num("checkpoint_interval", integer=True),
                           sample_interval=num("sample_interval", integer=True),
                           statistics_save_interval=num("statistics_save_interval", integer=True)),
        )
        if params["recording"]["statistics_save_interval"] % params["recording"]["sample_interval"]:
            raise ValueError("Statistics save interval must be a multiple of the observable sampling interval.")
        if mode == "L-FTS":
            if len(params["segment_lengths"]) == 2 and next(iter(params["chi_n"].values())) <= 0:
                raise ValueError("This A/B L-FTS setup requires positive chi N.")
            mixing["name"] = choice("compressor", ("am", "lr", "lram"))
            # The library's rule (lfts.py): LR needs the pressure field as the only imaginary one
            if mixing["name"] != "am" and imaginary_exchange_fields(list(params["segment_lengths"]), params["chi_n"]):
                raise ValueError("Compressors lr and lram need the pressure field as the only imaginary field; "
                                 "these chi N values give another one. Choose am.")
            params.update(saddle=dict(max_iter=num("saddle_max_iter", integer=True),
                                      tolerance=num("saddle_tolerance")), compressor=mixing)
        else:
            for key in ("zeta_n", "dynamic_stabilization"):
                params[key] = num(key, positive=False)
                if params[key] < 0:
                    raise ValueError(f"{key}: enter a non-negative number.")
            smearing = choice("smearing", ("none", "gaussian", "sigmoidal"))
            if smearing == "gaussian":
                a_int = num("a_int", positive=False)
                if a_int < 0:
                    raise ValueError("a_int: enter a non-negative number.")
                params["smearing"] = dict(type=smearing, a_int=a_int)
            elif smearing == "sigmoidal":
                params["smearing"] = dict(type=smearing, k_int=num("k_int"), dk_int=num("dk_int"))

    names = list(params["segment_lengths"])
    if len(names) > 2 and initial in ("lamella", "gyroid"):
        raise ValueError("Lamella and gyroid seeds require two monomer types. Choose random or mat for larger systems.")
    if initial == "mat":
        if mode == "CL-FTS":
            read_field = dedent('''\
                if key in field_data:
                    field = np.asarray(field_data[key], dtype=np.complex128)
                elif key + "_real" in field_data and key + "_imag" in field_data:
                    real = np.asarray(field_data[key + "_real"])
                    imag = np.asarray(field_data[key + "_imag"])
                    if real.shape != imag.shape or np.iscomplexobj(real) or np.iscomplexobj(imag):
                        raise ValueError(f"{key}_real and {key}_imag must be real arrays of the same shape.")
                    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
                        raise ValueError(f"{key} must contain one finite value per grid point.")
                    field = real.astype(np.float64) + 1j * imag.astype(np.float64)
                else:
                    raise ValueError(f"Initial .mat file needs {key} or both {key}_real and {key}_imag.")
                ''')
        else:
            read_field = dedent('''\
                if key not in field_data:
                    raise ValueError(f"Initial .mat file is missing {key}.")
                if np.iscomplexobj(field_data[key]):
                    raise ValueError(f"{key} must be real-valued for SCFT/L-FTS.")
                field = np.asarray(field_data[key], dtype=np.float64)
                ''')
        field_code = dedent(f"""\
            # Relative input paths are resolved next to this script.
            from scipy.io import loadmat

            field_data = loadmat(script_dir / {mat_file!r}, squeeze_me=True)
            if "nx" in field_data and not np.array_equal(np.atleast_1d(field_data["nx"]), params["nx"]):
                raise ValueError("The .mat grid (nx) does not match params['nx'].")
            initial_fields = {{}}
            for monomer in params["segment_lengths"]:
                key = "w_" + monomer
            """)
        field_code += indent(read_field + dedent('''\
            if field.size != math.prod(params["nx"]) or not np.all(np.isfinite(field)):
                raise ValueError(f"{key} must contain one finite value per grid point.")
            initial_fields[monomer] = field.reshape(params["nx"])
            '''), "    ")
    else:
        amplitude = num("amplitude")
        if initial == "lamella":
            field_code = dedent(f"""\
                # Cell-centered lamellar seed along the first axis.
                x = 2 * np.pi * (np.arange(params["nx"][0]) + 0.5) / params["nx"][0]
                shape = [params["nx"][0]] + [1] * (len(params["nx"]) - 1)
                w_a = {amplitude!r} * np.broadcast_to(np.cos(x).reshape(shape), params["nx"]).copy()
                """)
        elif initial == "gyroid":
            field_code = dedent(f"""\
                # Gyroid seed from examples/block_copolymer/scft/Gyroid.py, on cell-centered grids.
                axes = [2 * np.pi * (np.arange(n) + 0.5) / n for n in params["nx"]]
                x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
                c1 = np.sqrt(8 / 3) * (np.cos(x) * np.sin(y) * np.sin(2 * z)
                     + np.cos(y) * np.sin(z) * np.sin(2 * x)
                     + np.cos(z) * np.sin(x) * np.sin(2 * y))
                c2 = np.sqrt(4 / 3) * (np.cos(2 * x) * np.cos(2 * y)
                     + np.cos(2 * y) * np.cos(2 * z) + np.cos(2 * z) * np.cos(2 * x))
                w_a = {amplitude!r} * (-0.3164 * c1 + 0.1074 * c2)
                """)
        else:
            field_code = dedent(f"""\
                rng = np.random.default_rng({seed!r})
                w_a = rng.normal(0.0, {amplitude!r}, size=params["nx"])
                """)
        if len(names) == 2:
            field_code += f'initial_fields = {{{names[0]!r}: w_a, {names[1]!r}: -w_a}}\n'
        else:
            field_code = dedent(f"""\
                rng = np.random.default_rng({seed!r})
                initial_fields = {{name: rng.normal(0.0, {amplitude!r}, size=params["nx"])
                                  for name in params["segment_lengths"]}}
                mean_field = sum(initial_fields.values()) / len(initial_fields)
                for field in initial_fields.values():
                    field -= mean_field
                """)
    module, cls = {"SCFT": ("scft", "SCFT"), "L-FTS": ("lfts", "LFTS"),
                   "CL-FTS": ("clfts", "CLFTS")}[mode]
    constructor = f"{module}.{cls}(params=params" + (f", random_seed={seed!r}" if mode != "SCFT" else "") + ")"
    return (
        '#!/usr/bin/env python3\n'
        f'"""{mode} simulation generated by scripts/gui/simulation_gui.py.\n\n'
        'Review the parameters and submit long calculations through SLURM.\n'
        'Lengths and fields use the reference-chain units of polymerfts.\n'
        '"""\n\n'
        'import os\n\n'
        '# Set OpenMP before importing NumPy or polymerfts.\n'
        f'os.environ["OMP_NUM_THREADS"] = {str(threads)!r}\n'
        'os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"\n'
        'os.environ.setdefault("OMP_STACKSIZE", "1G")\n\n'
        'import math\n'
        'from pathlib import Path\n\n'
        'import numpy as np\n'
        f'from polymerfts import {module}\n\n\n'
        'params = ' + pformat(params, sort_dicts=False, width=92) + '\n\n\n'
        'def main():\n'
        '    script_dir = Path(__file__).resolve().parent\n'
        + indent(field_code, '    ') + '\n'
        '    # Keep architecture drawings and results together, away from the source tree.\n'
        f'    output_dir = script_dir / {output_dir!r}\n'
        '    output_dir.mkdir(parents=True, exist_ok=True)\n'
        '    os.chdir(output_dir)\n'
        f'    simulation = {constructor}\n'
        '    simulation.run(initial_fields=initial_fields)\n'
        + ('    simulation.save_results("fields.mat")\n' if mode == 'SCFT' else '')
        + '\n\nif __name__ == "__main__":\n    main()\n'
    )


# Directory descriptors keep every write inside the root; Windows has none
DIRECTORY_DESCRIPTORS = (os.open in os.supports_dir_fd and os.mkdir in os.supports_dir_fd
                         and hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"))


def save_script(script, root, directory, filename):
    """Create a new script under the configured root, without following links."""
    if not isinstance(directory, str) or not isinstance(filename, str):
        raise ValueError("Server folder and filename must be text.")
    filename = filename.strip()
    folder = Path(directory.strip())
    if len(directory) > 512 or len(folder.parts) > 16:
        raise ValueError("Server subfolder must have at most 512 characters and 16 path components.")
    # anchor: a root or a drive. On Windows "/tmp" is not absolute but rooted on the current
    # drive, and ':' separates a drive ("C:x") or names an alternate data stream
    if (folder.anchor or ".." in folder.parts or "\\" in directory or ":" in directory
            or any(ord(char) < 32 for char in directory)):
        raise ValueError("Server subfolder must be a relative path without '..'.")
    if (not filename.lower().endswith(".py") or "/" in filename or "\\" in filename or ":" in filename
            or any(ord(char) < 32 for char in filename) or len(filename.encode("utf-8")) > 240):
        raise ValueError("Enter a .py filename without path separators.")
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    except FileExistsError as exc:
        raise NotADirectoryError(f"Server script directory is not a directory: {root}") from exc
    if not DIRECTORY_DESCRIPTORS:
        return save_script_by_path(script, root, folder, filename)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(root, flags)
    try:
        for part in folder.parts:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            # Directory descriptors keep writes inside the selected root even if a
            # path component is changed to a symlink during another request.
            child_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        file_fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                          mode=0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(file_fd, "w", encoding="utf-8", newline="\n") as output:
                output.write(script)
        except OSError:
            os.unlink(filename, dir_fd=directory_fd)
            raise
    finally:
        os.close(directory_fd)
    return root / folder / filename


def save_script_by_path(script, root, folder, filename):
    """Windows: create the folders one at a time and refuse links (symlinks, junctions).

    Without directory descriptors, a folder that another process replaces by a link
    between this check and the write is not caught.
    """
    def refuse_link(path):
        # A link resolves to another path
        if not path.is_dir() or os.path.normcase(path.resolve(strict=True)) != os.path.normcase(path.absolute()):
            raise OSError(f"Server subfolder is a link or not a folder: {path}")

    path = root
    refuse_link(path)
    for part in folder.parts:
        path = path / part
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        refuse_link(path)
    # O_EXCL also refuses an existing link; O_BINARY keeps the line endings of the script
    target = path / filename
    file_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        with os.fdopen(file_fd, "w", encoding="utf-8", newline="\n") as output:
            output.write(script)
    except OSError:
        os.unlink(target)
        raise
    return target


class ScriptRequestHandler(BaseHTTPRequestHandler):
    """Serve editor assets, generate scripts, and save new scripts under one root."""

    def log_message(self, *_):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def respond(self, status, body, content_type):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; "
                             "base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A browser may cancel a superseded preview request.
            pass

    def respond_json(self, status, data):
        self.respond(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                     "application/json; charset=utf-8")

    def check_origin(self):
        host = self.headers.get("Host", "")
        if not re.fullmatch(r"(?:127\.0\.0\.1|localhost)(?::[0-9]{1,5})?", host, re.IGNORECASE):
            self.respond_json(403, {"error": "Connect through localhost or 127.0.0.1."})
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + host:
            self.respond_json(403, {"error": "Cross-origin requests are not accepted."})
            return False
        # Loopback is shared by every account on the machine: the API also needs the
        # token of the editor's URL, which only the user who started it sees
        # (as bytes: compare_digest refuses str with non-ASCII characters)
        if urlsplit(self.path).path.startswith("/api/") and not hmac.compare_digest(
                self.headers.get("X-Editor-Token", "").encode("latin-1", "replace"), self.server.token.encode()):
            self.respond_json(403, {"error": "Open the editor with the URL (including its token) that the server printed."})
            return False
        return True

    def do_GET(self):
        if not self.check_origin():
            return
        route = urlsplit(self.path).path
        if route == "/api/presets":
            self.respond_json(200, {name: preset_settings(name) for name in PRESETS})
            return
        if route == "/api/config":
            try:
                self.respond_json(200, {"save_dir": str(self.server.save_dir), "space_groups": space_group_options()})
            except (OSError, ValueError):
                self.respond_json(500, {"error": "Space group catalog is missing or invalid. Keep assets/ next to simulation_gui.py."})
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/model-editor.js": ("model-editor.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8"),
                  "/icon.svg": ("icon.svg", "image/svg+xml")}
        if route not in assets:
            self.respond_json(404, {"error": "Not found."})
            return
        filename, content_type = assets[route]
        try:
            body = (Path(__file__).parent / "assets" / filename).read_bytes()
        except OSError:
            self.respond_json(500, {"error": "Editor assets are missing. Keep assets/ next to simulation_gui.py."})
            return
        self.respond(200, body, content_type)

    def do_POST(self):
        if not self.check_origin():
            return
        route = urlsplit(self.path).path
        if route not in ("/api/generate", "/api/save"):
            self.respond_json(404, {"error": "Not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self.respond_json(415, {"error": "Send settings as application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Negative body length")
        except ValueError:
            self.respond_json(400, {"error": "Invalid Content-Length."})
            return
        if length > 65536:
            self.respond_json(413, {"error": "Settings must be smaller than 64 KiB."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if route == "/api/save":
                if not isinstance(payload, dict) or payload.keys() != {"values", "filename", "directory"}:
                    raise ValueError("Send settings, filename and server subfolder.")
                values = payload["values"]
            else:
                values = payload
            defaults = preset_settings(PRESETS[0])
            if isinstance(values, dict) and "model" in values:
                for key in ("nx", "lx", "blocks", "b_a", "b_b", "chi_n"):
                    defaults.pop(key)
                defaults["model"] = {}
            if not isinstance(values, dict) or values.keys() != defaults.keys():
                raise ValueError("Send the complete settings form with no extra fields.")
            for key, default in defaults.items():
                if type(values[key]) is not type(default):
                    raise ValueError(f"{key}: expected {type(default).__name__}.")
            script = build_script(values)
            if route == "/api/save":
                path = save_script(script, self.server.save_dir, payload["directory"], payload["filename"])
                self.respond_json(201, {"path": str(path), "script": script})
                return
        except (ValueError, RecursionError) as exc:
            self.respond_json(400, {"error": str(exc)})
            return
        except TimeoutError:
            self.respond_json(408, {"error": "Timed out reading settings."})
            return
        except FileExistsError:
            self.respond_json(409, {"error": "The file already exists. Choose another filename; existing files are not overwritten."})
            return
        except OSError as exc:
            self.respond_json(500, {"error": f"Could not save the script on the server: {exc}"})
            return
        self.respond_json(200, {"script": script})


def create_server(port=8765, save_dir=None):
    """Bind to loopback only; SSH forwarding supplies remote browser access. The
    API requires the random token of server.token, which the printed URL carries."""
    save_dir = Path(save_dir if save_dir is not None else Path.cwd()).expanduser().resolve()
    server = ThreadingHTTPServer(("127.0.0.1", port), ScriptRequestHandler)
    server.save_dir = save_dir
    server.token = secrets.token_urlsafe(32)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Local HTTP port (default: 8765)")
    parser.add_argument("--save-dir", type=Path,
                        help="Server script directory (default: the current working directory)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = create_server(args.port, args.save_dir)
    except OSError as exc:
        parser.exit(1, f"Could not start editor: {exc}. Check --port and --save-dir.\n")
    with server:
        port = server.server_address[1]
        print(f"Simulation Script Builder (beta): http://127.0.0.1:{port}/?token={server.token}", flush=True)
        print(f"Server script directory: {server.save_dir}", flush=True)
        print(f"SSH tunnel (on your computer): ssh -L {port}:127.0.0.1:{port} user@server", flush=True)
        print("Press Ctrl+C to stop. This editor only generates scripts.", flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            print("\nEditor stopped.", flush=True)


if __name__ == "__main__":
    main()
