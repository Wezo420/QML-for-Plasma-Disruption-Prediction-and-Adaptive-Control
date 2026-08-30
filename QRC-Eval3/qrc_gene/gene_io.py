"""
gene_io.py
==========
Readers for GENE (gyrokinetic turbulence code) simulation output.

GENE writes several diagnostic files per run. This module knows how to
parse the ones needed to build a QRC-ready time series:

    parameters / parameters.dat   Fortran namelists describing the run
                                   (grid sizes, species, geometry, ...)
    nrg.dat                       ASCII time trace of spatially-averaged
                                   fluctuation amplitudes and fluxes,
                                   one block of `n_spec` lines per step.
    energy.dat                    ASCII time trace of the free-energy
                                   budget (14 columns, see GENE manual
                                   Sec. 4.6).
    circular.dat (or similar)     ASCII geometry file: a namelist header
                                   followed by one row per parallel (z)
                                   grid point.
    field.dat / mom_<species>.dat Binary (Fortran unformatted, sequential)
                                   3D (kx, ky, z) complex snapshots of the
                                   electrostatic potential / distribution
                                   function moments (GENE manual Sec. 4.2-4.3).

All readers are written to be robust to the fact that in practice you may
only have a *chunk* of a field/mom file (these files can run into the
hundreds of MB for a real nonlinear run and are often shipped as partial
byte ranges for inspection). The binary reader below reads as many
*complete* records as are present and stops cleanly instead of raising an
error on a truncated final record.

Column/field definitions here are taken from the GENE User Manual
(release 3.1), Sec. 4.1-4.3 -- not guessed from the data.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Namelist (parameters / parameters.dat) parsing
# --------------------------------------------------------------------------

_NML_HEADER = re.compile(r"^&(\w+)\s*$")
_NML_FOOTER = re.compile(r"^/\s*$")
# key = value   (value may be a fortran logical .t./.f., a quoted string,
# a number in fortran D/E exponent form, or a whitespace separated list)
_NML_ENTRY = re.compile(r"^\s*([A-Za-z_][\w]*)\s*=\s*(.*?)\s*$")


def _coerce_scalar(token: str):
    """Convert a single Fortran-namelist token to a Python type."""
    token = token.strip()
    if token in (".t.", ".T.", "T", "t", ".true.", ".TRUE."):
        return True
    if token in (".f.", ".F.", "F", "f", ".false.", ".FALSE."):
        return False
    if len(token) >= 2 and token[0] == "'" and token[-1] == "'":
        return token[1:-1]
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    # Fortran doubles sometimes use D instead of E for the exponent
    fortran_float = token.replace("D", "E").replace("d", "e")
    try:
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
        return float(fortran_float)
    except ValueError:
        return token  # leave as string (e.g. unrecognised token)


def _coerce_value(raw: str):
    raw = raw.split("!")[0].strip()  # strip trailing fortran comments
    if raw == "":
        return None
    parts = raw.split()
    if len(parts) == 1:
        return _coerce_scalar(parts[0])
    # multiple whitespace-separated tokens -> list (e.g. perf_vec = 1 1 1 1)
    return [_coerce_scalar(p) for p in parts]


def parse_gene_namelist(path: str | Path) -> Dict[str, Dict[str, object]]:
    """Parse a GENE ``parameters`` / ``parameters.dat`` file.

    Returns a nested dict: ``{namelist_name: {key: value}}``, e.g.
    ``params['box']['nx0'] -> 96``. Works for both the pre-run template
    (``parameters``) and the post-run output (``parameters.dat``), which
    additionally contains an ``&info`` block with the actual grid sizes
    used (``n_fields``, ``n_moms``, ``nrgcols``) -- these are exactly the
    quantities needed to read the binary field/mom files correctly, so
    prefer parsing ``parameters.dat`` when it is available.
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    namelists: Dict[str, Dict[str, object]] = {}
    current = None
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        m = _NML_HEADER.match(line.strip())
        if m:
            current = m.group(1).lower()
            namelists[current] = {}
            continue
        if _NML_FOOTER.match(line.strip()):
            current = None
            continue
        if current is None:
            continue
        m = _NML_ENTRY.match(line)
        if m:
            key, raw_val = m.group(1).lower(), m.group(2)
            namelists[current][key] = _coerce_value(raw_val)
    return namelists


@dataclass
class GeneRunInfo:
    """Convenience summary of the run dimensions needed by the readers
    below, extracted from a parsed namelist dict."""

    nx0: int
    nky0: int
    nz0: int
    n_spec: int
    n_fields: int
    n_moms: int
    nrgcols: int
    species_names: List[str] = field(default_factory=list)
    dt_max: Optional[float] = None
    raw: Dict[str, Dict[str, object]] = field(default_factory=dict)

    @classmethod
    def from_namelists(cls, nml: Dict[str, Dict[str, object]]) -> "GeneRunInfo":
        box = nml.get("box", {})
        info = nml.get("info", {})
        general = nml.get("general", {})
        species = nml.get("species", {})

        nx0 = int(box.get("nx0"))
        nky0 = int(box.get("nky0"))
        nz0 = int(box.get("nz0"))
        n_spec = int(box.get("n_spec", 1))

        # n_fields / n_moms / nrgcols only appear in the post-run
        # parameters.dat &info block. Fall back to sensible GENE defaults
        # (electrostatic, 6 moments, 8 nrg columns) if only the pre-run
        # template is available.
        n_fields = int(info.get("n_fields", 1))
        n_moms = int(info.get("n_moms", 6))
        nrgcols = int(info.get("nrgcols", 8))

        species_names = [species.get("name", "ions")] if species else ["ions"]

        dt_max = general.get("dt_max")

        return cls(
            nx0=nx0, nky0=nky0, nz0=nz0, n_spec=n_spec,
            n_fields=n_fields, n_moms=n_moms, nrgcols=nrgcols,
            species_names=species_names, dt_max=dt_max, raw=nml,
        )


def load_run_info(parameters_path: str | Path) -> GeneRunInfo:
    nml = parse_gene_namelist(parameters_path)
    return GeneRunInfo.from_namelists(nml)


# --------------------------------------------------------------------------
# nrg.dat
# --------------------------------------------------------------------------

# Standard 8-column nrg block (GENE manual Sec. 4.1). Two more columns are
# appended when tor_ang_mom_flux = T.
NRG_COLUMNS_8 = [
    "n1_sq", "u1par_sq", "T1par_sq", "T1perp_sq",
    "Gamma_es", "Gamma_em", "Q_es", "Q_em",
]
NRG_COLUMNS_10 = NRG_COLUMNS_8 + ["Pi_es", "Pi_em"]


def read_nrg(path: str | Path, n_spec: int = 1, nrgcols: Optional[int] = None):
    """Read a GENE ``nrg.dat`` file.

    Args:
        path: path to nrg.dat
        n_spec: number of species (number of data lines per time block)
        nrgcols: number of columns per species line (8 or 10). If None,
            it is inferred from the file itself.

    Returns:
        time:  (n_t,) array of simulation times
        data:  (n_t, n_spec, n_cols) array
        columns: list of column names of length n_cols
    """
    path = Path(path)
    lines = [l for l in path.read_text().splitlines() if l.strip()]

    if nrgcols is None:
        # peek at the first data line to infer the column count
        first_data_line = lines[1]
        nrgcols = len(first_data_line.split())

    columns = NRG_COLUMNS_10 if nrgcols >= 10 else NRG_COLUMNS_8
    columns = columns[:nrgcols]

    n_lines_per_block = 1 + n_spec
    n_blocks = len(lines) // n_lines_per_block

    time = np.empty(n_blocks)
    data = np.empty((n_blocks, n_spec, nrgcols))

    for b in range(n_blocks):
        base = b * n_lines_per_block
        time[b] = float(lines[base])
        for s in range(n_spec):
            row = lines[base + 1 + s].split()
            data[b, s, :] = [float(x) for x in row[:nrgcols]]

    return time, data, columns


# --------------------------------------------------------------------------
# energy.dat
# --------------------------------------------------------------------------

ENERGY_COLUMNS = [
    "time", "Etot", "dEdt_tot", "dEdt_drive", "dEdt_source", "dEdt_coll",
    "dEdt_Dz", "dEdt_Dv", "dEdt_Dxy", "dEdt_nl", "dEdt_zv", "dEdt_rest",
    "dEdt_check", "dEdt_tot2",
]


def read_energy(path: str | Path) -> Dict[str, np.ndarray]:
    """Read a GENE ``energy.dat`` file (14 columns, see GENE manual
    Sec. 4.6). Lines starting with ``#`` are header/comment lines and are
    skipped. Returns a dict mapping column name -> 1D array."""
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    arr = np.array(rows)
    ncols = min(arr.shape[1], len(ENERGY_COLUMNS))
    return {ENERGY_COLUMNS[i]: arr[:, i] for i in range(ncols)}


# --------------------------------------------------------------------------
# geometry file (e.g. circular.dat)
# --------------------------------------------------------------------------

def read_geometry(path: str | Path):
    """Read a GENE magnetic-geometry output file (namelist header +
    one row per z grid point). Returns (params_dict, data_array)."""
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    # namelist header up to the closing '/'
    header_lines, i = [], 0
    for i, line in enumerate(lines):
        header_lines.append(line)
        if line.strip() == "/":
            break
    header_text = "\n".join(header_lines)
    params = parse_gene_namelist_from_text(header_text)

    data_rows = []
    for line in lines[i + 1:]:
        if line.strip():
            data_rows.append([float(x) for x in line.split()])
    data = np.array(data_rows)
    return params, data


def parse_gene_namelist_from_text(text: str) -> Dict[str, object]:
    """Like parse_gene_namelist but for an in-memory string containing a
    single (possibly unlabeled) namelist block, used by read_geometry."""
    out: Dict[str, object] = {}
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip() or line.strip().startswith("&") or line.strip() == "/":
            continue
        m = _NML_ENTRY.match(line)
        if m:
            out[m.group(1).lower()] = _coerce_value(m.group(2))
    return out


# --------------------------------------------------------------------------
# Binary field.dat / mom_<species>.dat reader
# --------------------------------------------------------------------------

@dataclass
class BinarySnapshots:
    times: np.ndarray               # (n_snap,)
    data: np.ndarray                # (n_snap, n_vars, nx0, nky0, nz0) complex128
    var_names: List[str]
    n_complete: int                 # number of fully-read snapshots
    truncated: bool                 # True if the file ended mid-record


def _read_fortran_record(f, marker_bytes: int) -> Optional[bytes]:
    """Read one Fortran unformatted sequential record (leading + trailing
    length markers of ``marker_bytes`` bytes each). Returns the payload,
    or None if fewer than a full record remains in the file."""
    head = f.read(marker_bytes)
    if len(head) < marker_bytes:
        return None
    fmt = "<i" if marker_bytes == 4 else "<q"
    (reclen,) = struct.unpack(fmt, head)
    payload = f.read(reclen)
    if len(payload) < reclen:
        return None
    tail = f.read(marker_bytes)
    if len(tail) < marker_bytes:
        return None
    return payload


def _detect_marker_bytes(f) -> int:
    """GENE binary files are Fortran unformatted sequential files. The
    very first record is always the (8-byte double) simulation time, so
    the opening marker must decode to the value 8. Try 4-byte markers
    (the default almost everywhere) and fall back to 8-byte markers."""
    pos = f.tell()
    head4 = f.read(4)
    f.seek(pos)
    if len(head4) == 4 and struct.unpack("<i", head4)[0] == 8:
        return 4
    head8 = f.read(8)
    f.seek(pos)
    if len(head8) == 8 and struct.unpack("<q", head8)[0] == 8:
        return 8
    # default assumption
    return 4


def read_gene_binary(
    path: str | Path,
    nx0: int,
    nky0: int,
    nz0: int,
    var_names: Sequence[str],
    max_snapshots: Optional[int] = None,
) -> BinarySnapshots:
    """Read a GENE binary diagnostic file (``field.dat`` or
    ``mom_<species>.dat``), including partial/truncated chunks.

    Each snapshot on disk is: [time record] + [one record per variable in
    ``var_names``, each holding nx0*nky0*nz0 complex128 numbers in
    (kx, ky, z) Fortran (column-major) order].

    For ``field.dat`` typical ``var_names`` are a subset of
    ``["phi", "Apar", "Bpar"]`` (length = n_fields).
    For ``mom_<species>.dat`` typical ``var_names`` are
    ``["n1", "T1par", "T1perp", "q1par", "q1perp", "u1par"]`` (length =
    n_moms, GENE manual Sec. 4.3).

    If the file is a truncated chunk (e.g. only the first few MB of a
    much larger run), this function reads every *complete* snapshot it
    can and reports how many via ``BinarySnapshots.n_complete`` /
    ``.truncated`` rather than raising an error.
    """
    path = Path(path)
    n_vars = len(var_names)
    n_per_var = nx0 * nky0 * nz0

    times: List[float] = []
    arrays: List[np.ndarray] = []
    truncated = False

    with open(path, "rb") as f:
        marker_bytes = _detect_marker_bytes(f)

        while True:
            if max_snapshots is not None and len(times) >= max_snapshots:
                break

            start_pos = f.tell()
            time_payload = _read_fortran_record(f, marker_bytes)
            if time_payload is None:
                break  # clean EOF (or truncated) at a snapshot boundary
            (t,) = struct.unpack("<d", time_payload[:8])

            snap_vars = []
            ok = True
            for _ in range(n_vars):
                payload = _read_fortran_record(f, marker_bytes)
                if payload is None:
                    ok = False
                    break
                arr = np.frombuffer(payload, dtype="<c16", count=n_per_var)
                arr = arr.reshape((nz0, nky0, nx0)).transpose(2, 1, 0)
                snap_vars.append(arr)

            if not ok:
                truncated = True
                f.seek(start_pos)  # rewind: this snapshot was incomplete
                break

            times.append(t)
            arrays.append(np.stack(snap_vars, axis=0))

        # if we stopped exactly on a snapshot boundary but there are still
        # leftover bytes (< 1 record), the file is a truncated chunk too
        remaining = f.read(1)
        if remaining:
            truncated = True

    if arrays:
        data = np.stack(arrays, axis=0)
        times_arr = np.array(times)
    else:
        data = np.empty((0, n_vars, nx0, nky0, nz0), dtype="complex128")
        times_arr = np.empty((0,))

    return BinarySnapshots(
        times=times_arr, data=data, var_names=list(var_names),
        n_complete=len(arrays), truncated=truncated,
    )


FIELD_VAR_NAMES_ALL = ["phi", "Apar", "Bpar"]
MOM_VAR_NAMES_ALL = ["n1", "T1par", "T1perp", "q1par_1p5p0u1par",
                      "q1perp_p0u1par", "u1par"]


def read_field_file(path, run_info: GeneRunInfo, max_snapshots=None) -> BinarySnapshots:
    names = FIELD_VAR_NAMES_ALL[: run_info.n_fields]
    return read_gene_binary(path, run_info.nx0, run_info.nky0, run_info.nz0,
                             names, max_snapshots=max_snapshots)


def read_mom_file(path, run_info: GeneRunInfo, max_snapshots=None) -> BinarySnapshots:
    names = MOM_VAR_NAMES_ALL[: run_info.n_moms]
    return read_gene_binary(path, run_info.nx0, run_info.nky0, run_info.nz0,
                             names, max_snapshots=max_snapshots)


# --------------------------------------------------------------------------
# Chunk diagnostics / concatenation (Eval-3 addition)
# --------------------------------------------------------------------------

def snapshot_byte_requirements(run_info: GeneRunInfo, n_vars: int, marker_bytes: int = 4) -> Dict[str, int]:
    """Exact byte accounting for one complete field/mom snapshot under
    ``run_info``'s grid, for diagnosing why a chunk file does or doesn't
    contain any complete snapshots (rather than guessing). A snapshot on
    disk is one time record (an 8-byte double) plus one record per
    variable (``nx0*nky0*nz0`` complex128 numbers each), each record
    wrapped in leading+trailing ``marker_bytes`` length markers."""
    n_per_var = run_info.nx0 * run_info.nky0 * run_info.nz0
    var_payload_bytes = n_per_var * 16  # complex128 = 16 bytes
    var_record_bytes = var_payload_bytes + 2 * marker_bytes
    time_record_bytes = 8 + 2 * marker_bytes
    return {
        "n_per_var": n_per_var,
        "var_payload_bytes": var_payload_bytes,
        "var_record_bytes": var_record_bytes,
        "time_record_bytes": time_record_bytes,
        "snapshot_bytes": time_record_bytes + n_vars * var_record_bytes,
    }


def concat_chunks(paths: Sequence[str | Path], out_path: str | Path) -> Path:
    """Concatenate several sequentially-numbered chunk files
    (``field_chunk_1.dat``, ``field_chunk_2.dat``, ...) into one file at
    ``out_path``, in the order given, so ``read_gene_binary`` can be run
    against the combined byte stream. Each chunk is assumed to be a raw
    contiguous byte range of the original file with no chunk-specific
    header of its own (i.e. only the *first* chunk starts at a snapshot
    boundary; this is a straight byte concatenation, not snapshot-aware
    merging). Useful once more chunks of ``field.dat`` / ``mom_ions.dat``
    become available than the single first chunk used during development
    of this module (see ``preprocessing.build_field_mom_features``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out_f:
        for p in paths:
            with open(p, "rb") as in_f:
                out_f.write(in_f.read())
    return out_path
