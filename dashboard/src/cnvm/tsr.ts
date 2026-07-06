export interface SlidersConfigEntry {
  name: string;
  description: string;
  // block_size is intentionally removed — every slider is exactly one index
}

export type SlidersConfig = Record<string, SlidersConfigEntry>;

export interface TsrMapInfo {
  tsrMap: Record<string, [number, number]>;
  dim: number;
}

/**
 * Builds the flat TSR coordinate map from sliders.json.
 *
 * Flat single-slot design: every slider occupies exactly one index.
 *   index = order in slidersConfig
 *   dim   = total number of entries
 *
 * No block_size, no stride, no padding. Adding a new concept = adding one entry.
 */
export function loadSliderMap(slidersConfig: SlidersConfig): TsrMapInfo {
  const tsrMap: Record<string, [number, number]> = {};
  let coord = 0;

  for (const [name, entry] of Object.entries(slidersConfig)) {
    // Single slot: start == end == coordinate
    tsrMap[name] = [coord, coord];
    coord++;
  }

  const dim = coord;
  return { tsrMap, dim };
}

export function createEmptyState(dim: number): number[] {
  return new Array(dim).fill(0);
}

export function setRegisterValue(
  state: number[],
  tsrMap: Record<string, [number, number]>,
  name: string,
  value: number
): void {
  if (!(name in tsrMap)) {
    throw new Error(`Unknown register name: ${name}`);
  }
  const [idx] = tsrMap[name];
  state[idx] = value;
}

export function getRegisterValue(
  state: number[],
  tsrMap: Record<string, [number, number]>,
  name: string
): number {
  if (!(name in tsrMap)) {
    throw new Error(`Unknown register name: ${name}`);
  }
  const [idx] = tsrMap[name];
  return state[idx];
}

export function slidersToVector(
  sliders: Record<string, number>,
  tsrMap: Record<string, [number, number]>,
  dim: number
): number[] {
  const state = createEmptyState(dim);
  for (const [name, val] of Object.entries(sliders)) {
    if (name in tsrMap) {
      setRegisterValue(state, tsrMap, name, val);
    }
  }
  return state;
}

export function vectorToSliders(
  state: number[],
  tsrMap: Record<string, [number, number]>
): Record<string, number> {
  const sliders: Record<string, number> = {};
  const metaNames = new Set(["META::PROVENANCE", "META::EVIDENCE", "RESERVED"]);
  for (const name of Object.keys(tsrMap)) {
    if (!metaNames.has(name)) {
      sliders[name] = getRegisterValue(state, tsrMap, name);
    }
  }
  return sliders;
}

/**
 * Returns the set of register indices that are 'semantic' (included in
 * layer normalisation and clipping). Excludes META::PROVENANCE and META::EVIDENCE.
 */
export function getSemanticIndices(
  tsrMap: Record<string, [number, number]>,
  dim: number
): number[] {
  const metaNames = new Set(["META::PROVENANCE", "META::EVIDENCE", "RESERVED"]);
  const metaIndices = new Set<number>();

  for (const [name, [start, end]] of Object.entries(tsrMap)) {
    if (metaNames.has(name)) {
      for (let i = start; i <= end; i++) {
        metaIndices.add(i);
      }
    }
  }

  const result: number[] = [];
  for (let i = 0; i < dim; i++) {
    if (!metaIndices.has(i)) {
      result.push(i);
    }
  }
  return result;
}
