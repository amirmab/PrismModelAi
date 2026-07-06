import json
import os
import numpy as np

# Lazy-loaded global state (populated by load_slider_map)
_TSR_MAP: dict[str, tuple[int, int]] = {}
_DIM: int = 0


def load_slider_map(manifest_dir: str = "manifest") -> dict[str, tuple[int, int]]:
    """
    Reads sliders.json and builds the flat TSR coordinate map.

    Each slider entry must have:
      - "coordinate": N  (unique integer, 0-based, determines index in state vector)

    Every slider occupies exactly ONE index in the state vector:
      index = coordinate

    This is the flat single-slot design — no block_size, no stride, no padding.
    To add a new concept, add a new entry with the next available coordinate.

    Returns:
        TSR_MAP: dict mapping slider_name -> (index, index) — both values equal
    """
    path = os.path.join(manifest_dir, "sliders.json")
    with open(path, "r") as f:
        data = json.load(f)

    tsr_map: dict[str, tuple[int, int]] = {}
    for coord, (name, entry) in enumerate(data.items()):
        # Each slider is a single slot: start == end == coord
        tsr_map[name] = (coord, coord)

    global _TSR_MAP, _DIM
    _TSR_MAP = tsr_map
    # DIM = total number of addressable registers = length of map
    _DIM = len(tsr_map)

    return tsr_map


def get_tsr_map() -> dict[str, tuple[int, int]]:
    """Returns the currently loaded TSR map. Must call load_slider_map() first."""
    if not _TSR_MAP:
        raise RuntimeError("TSR map not loaded. Call load_slider_map(manifest_dir) first.")
    return _TSR_MAP


def get_dim() -> int:
    """Returns the total hidden dimension (= number of sliders). Must call load_slider_map() first."""
    if _DIM == 0:
        raise RuntimeError("TSR map not loaded. Call load_slider_map(manifest_dir) first.")
    return _DIM


def get_register_offset(name: str) -> int:
    """Returns the index of a named register in the state vector."""
    tsr_map = get_tsr_map()
    if name not in tsr_map:
        raise ValueError(f"Unknown register name: '{name}'. Available: {list(tsr_map.keys())}")
    return tsr_map[name][0]


def get_register_slice(name: str) -> slice:
    """Returns a slice covering this register's index (always a single element)."""
    tsr_map = get_tsr_map()
    if name not in tsr_map:
        raise ValueError(f"Unknown register name: '{name}'")
    idx, _ = tsr_map[name]
    return slice(idx, idx + 1)


def create_empty_state() -> np.ndarray:
    """Returns a zero-initialized state vector of size DIM."""
    return np.zeros(get_dim(), dtype=np.float32)


def set_register_value(state: np.ndarray, name: str, value: float) -> None:
    """Sets the value of a named register in the state vector."""
    idx = get_register_offset(name)
    state[idx] = value


def get_register_value(state: np.ndarray, name: str) -> float:
    """Gets the value of a named register from the state vector."""
    idx = get_register_offset(name)
    return float(state[idx])


def sliders_to_vector(sliders: dict[str, float]) -> np.ndarray:
    """Converts a dict of {slider_name: value} into a DIM-dimensional register vector."""
    state = create_empty_state()
    for name, value in sliders.items():
        if name in get_tsr_map():
            set_register_value(state, name, float(value))
        else:
            raise ValueError(f"Slider '{name}' does not correspond to a known register.")
    return state


def vector_to_sliders(state: np.ndarray) -> dict[str, float]:
    """Extracts all named register values from a state vector."""
    tsr_map = get_tsr_map()
    return {
        name: get_register_value(state, name)
        for name in tsr_map
        if name not in ("META::PROVENANCE", "META::EVIDENCE")
    }


def format_state(state: np.ndarray) -> str:
    """Returns a clean string summary of active registers for logging/debugging."""
    tsr_map = get_tsr_map()
    lines = ["TSR Hidden State:"]
    for name, (idx, _) in tsr_map.items():
        val = state[idx]
        if abs(val) > 1e-5:
            lines.append(f"  {name:<35} [{idx:02d}]: {val:.4f}")
    return "\n".join(lines)
