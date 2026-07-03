export function clampWeight(val: number, limit = 2.0): number {
  return Math.max(Math.min(val, limit), -limit);
}

export interface RuleNir {
  rule_id: number;
  layer_index: number;
  trigger_slider_name: string;
  result_slider_name: string;
  gate_in_weight: number;
  gate_out_weight: number;
  intent_description?: string;
}

export interface VocabToken {
  token_id: number;
  concept_description?: string;
  sliders?: Record<string, number | { value: number; description?: string }>;
  inherits?: string;
  overrides?: Record<string, number | { value: number; description?: string }>;
}

export interface DomainRouting {
  bridges?: Record<string, number>;
}

export interface CompiledLayer {
  W_in: number[][];  // (DIM, R)
  W_out: number[][]; // (R, DIM)
  b: number[];       // (R)
  rule_ids: number[];
}

export interface CompiledCheckpoint {
  embedding: number[][]; // (V, DIM)
  token_to_id: Record<string, number>;
  id_to_token: Record<number, string>;
  W_q: number[][]; // (DIM, DIM)
  W_k: number[][]; // (DIM, DIM)
  W_v: number[][]; // (DIM, DIM)
  serg: Record<number, CompiledLayer>;
}
