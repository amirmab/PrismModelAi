import type { CompiledCheckpoint, CompiledLayer, ArchitectureConfig } from "./compiler_types";

type Vector = number[];
type Matrix = number[][];

export interface CceStepHistory {
  iteration: number;
  conflict_value: number;
  state: Matrix;
  active_rules: ActiveRule[][];
}

export interface CceInfo {
  converged: boolean;
  iterations: number;
  history: CceStepHistory[];
}

export interface ActiveRule {
  rule_id: number;
  activation: number;
  trigger_val: number;
}

export interface ExecutionTraceStep {
  layer: number;
  type: "STANDARD" | "CCE";
  active_rules?: ActiveRule[][];
  cce_info?: CceInfo;
}

export function dotProduct(v1: Vector, v2: Vector): number {
  let sum = 0;
  const len = v1.length;
  for (let i = 0; i < len; i++) {
    sum += v1[i] * v2[i];
  }
  return sum;
}

export function vectorMatrixMul(v: Vector, m: Matrix): Vector {
  const dOut = m[0].length;
  const dIn = v.length;
  const res = new Array(dOut).fill(0);
  for (let j = 0; j < dOut; j++) {
    let sum = 0;
    for (let i = 0; i < dIn; i++) {
      sum += v[i] * m[i][j];
    }
    res[j] = sum;
  }
  return res;
}

export function matrixMatrixMul(a: Matrix, b: Matrix): Matrix {
  return a.map(row => vectorMatrixMul(row, b));
}

export function getSemanticIndices(tsrMap: Record<string, [number, number]>, dim: number): number[] {
  const indices: number[] = [];
  const provRange = tsrMap["META::PROVENANCE"];
  const evRange = tsrMap["META::EVIDENCE"];
  const resRange = tsrMap["RESERVED"];

  const targetDim = 94; // Lock to baseline semantic dimension

  const isProv = (idx: number) => provRange && idx >= provRange[0] && idx <= provRange[1];
  const isEv = (idx: number) => evRange && idx >= evRange[0] && idx <= evRange[1];
  const isRes = (idx: number) => resRange && idx >= resRange[0] && idx <= resRange[1];

  for (let i = 0; i < targetDim; i++) {
    if (!isProv(i) && !isEv(i) && !isRes(i)) {
      indices.push(i);
    }
  }
  return indices;
}

export function layerNorm(H: Matrix, semanticIndices: number[], eps = 1e-5): Matrix {
  const N = H.length;
  const res: Matrix = H.map(row => [...row]);
  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (const idx of semanticIndices) {
      sum += res[i][idx];
    }
    const mean = sum / semanticIndices.length;

    let varSum = 0;
    for (const idx of semanticIndices) {
      const diff = res[i][idx] - mean;
      varSum += diff * diff;
    }
    const variance = varSum / semanticIndices.length;
    const std = Math.sqrt(variance + eps);

    for (const idx of semanticIndices) {
      res[i][idx] = (res[i][idx] - mean) / std;
    }
  }
  return res;
}

export function clipState(H: Matrix, semanticIndices: number[]): Matrix {
  const N = H.length;
  const res: Matrix = H.map(row => [...row]);
  const semanticSet = new Set(semanticIndices);

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < res[i].length; j++) {
      if (semanticSet.has(j)) {
        res[i][j] = Math.max(-2.0, Math.min(2.0, res[i][j]));
      }
    }
  }
  return res;
}

export class CNVMRuntime {
  embedding: number[][];
  token_to_id: Record<string, number>;
  W_q: number[][];
  W_k: number[][];
  W_v: number[][];
  serg: Record<number, CompiledLayer>;
  gamma = 1.0;
  semanticIndices: number[];
  conflictOffset: number;
  tsrMap: Record<string, [number, number]>;
  dim: number;
  architecture: ArchitectureConfig;

  constructor(
    compiledCheckpoint: CompiledCheckpoint,
    tsrMap: Record<string, [number, number]>,
    dim: number
  ) {
    this.embedding = compiledCheckpoint.embedding;
    this.token_to_id = compiledCheckpoint.token_to_id;
    this.W_q = compiledCheckpoint.W_q;
    this.W_k = compiledCheckpoint.W_k;
    this.W_v = compiledCheckpoint.W_v;
    this.serg = compiledCheckpoint.serg;
    this.architecture = compiledCheckpoint.architecture || {
      max_layers: 40,
      cce_layers: [30, 31, 32, 33, 34],
      default_cce_max_iter: 10,
      default_cce_epsilon: 0.1,
      layers: []
    };
    this.tsrMap = tsrMap;
    this.dim = dim;

    this.semanticIndices = getSemanticIndices(tsrMap, dim);
    this.conflictOffset = tsrMap["SYS::CONFLICT"] ? tsrMap["SYS::CONFLICT"][0] : 9;
  }

  getTokenVector(token: string): number[] {
    if (!(token in this.token_to_id)) {
      throw new Error(`Token '${token}' not found in compiled vocabulary.`);
    }
    const tokenId = this.token_to_id[token];
    return [...this.embedding[tokenId]];
  }

  executeAttention(H: Matrix): Matrix {
    const N = H.length;
    const Q = matrixMatrixMul(H, this.W_q);
    const K = matrixMatrixMul(H, this.W_k);
    const V = matrixMatrixMul(H, this.W_v);

    const logits: Matrix = Array.from({ length: N }, () => new Array(N).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let logit = dotProduct(Q[i], K[j]);
        if (N > 1) {
          const distance = j - i;
          
          // Asymmetric Bias: breaks N > M vs N < M symmetry
          const alibiBias = 0.1 * distance;
          logit += alibiBias;
          
          // Local Sliding Window Cutoff
          if (Math.abs(distance) > 5) {
            logit = -Infinity;
          }
        }
        logits[i][j] = logit;
      }
    }

    const attnWeights: Matrix = Array.from({ length: N }, () => new Array(N).fill(0));
    for (let i = 0; i < N; i++) {
      let maxVal = -Infinity;
      for (let j = 0; j < N; j++) {
        if (logits[i][j] > maxVal) maxVal = logits[i][j];
      }
      let sum = 0;
      const exps = new Array(N);
      for (let j = 0; j < N; j++) {
        let e = 0;
        if (logits[i][j] !== -Infinity) {
          e = Math.exp(logits[i][j] - maxVal);
        }
        exps[j] = e;
        sum += e;
      }
      for (let j = 0; j < N; j++) {
        attnWeights[i][j] = exps[j] / (sum || 1.0);
      }
    }

    const O: Matrix = Array.from({ length: N }, () => new Array(this.dim).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < this.dim; j++) {
        let sum = 0;
        for (let k = 0; k < N; k++) {
          sum += attnWeights[i][k] * V[k][j];
        }
        O[i][j] = sum;
      }
    }

    return O;
  }

  executeSergLayer(H: Matrix, layerIndex: number): { nextH: Matrix; activeRules: ActiveRule[][] } {
    if (!(layerIndex in this.serg)) {
      return { nextH: H.map(row => [...row]), activeRules: Array.from({ length: H.length }, () => []) };
    }

    const layerData = this.serg[layerIndex];
    const W_in = layerData.W_in;
    const W_out = layerData.W_out;
    const b = layerData.b;
    const ruleIds = layerData.rule_ids;

    const N = H.length;
    const R = b.length;

    const activations: Matrix = Array.from({ length: N }, () => new Array(R).fill(0));
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < R; c++) {
        let sum = b[c];
        for (let j = 0; j < this.dim; j++) {
          sum += H[i][j] * W_in[j][c];
        }
        activations[i][c] = Math.max(0.0, sum);
      }
    }

    const updates: Matrix = Array.from({ length: N }, () => new Array(this.dim).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < this.dim; j++) {
        let sum = 0;
        for (let c = 0; c < R; c++) {
          sum += activations[i][c] * W_out[c][j];
        }
        updates[i][j] = this.gamma * sum;
      }
    }

    const nextH: Matrix = Array.from({ length: N }, () => new Array(this.dim).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < this.dim; j++) {
        nextH[i][j] = H[i][j] + updates[i][j];
      }
    }

    const activeRules: ActiveRule[][] = [];
    for (let i = 0; i < N; i++) {
      const tokenActive: ActiveRule[] = [];
      for (let c = 0; c < R; c++) {
        const actVal = activations[i][c];
        if (actVal > 0) {
          let maxWeightIdx = 0;
          let maxAbsWeight = -1;
          for (let j = 0; j < this.dim; j++) {
            const absW = Math.abs(W_in[j][c]);
            if (absW > maxAbsWeight) {
              maxAbsWeight = absW;
              maxWeightIdx = j;
            }
          }

          tokenActive.push({
            rule_id: ruleIds[c],
            activation: actVal,
            trigger_val: H[i][maxWeightIdx]
          });
        }
      }
      activeRules.push(tokenActive);
    }

    return { nextH, activeRules };
  }

  executeCceLayer(H: Matrix, layerIndex: number, maxIter = 10, epsilon = 0.1): { hLayer: Matrix; cceInfo: CceInfo } {
    let hLayer = H.map(row => [...row]);
    const history: CceStepHistory[] = [];
    let iteration = 0;
    let converged = false;

    while (iteration < maxIter) {
      const hNorm = layerNorm(hLayer, this.semanticIndices);
      const attnOut = this.executeAttention(hNorm);
      
      let routed: Matrix = Array.from({ length: hLayer.length }, () => new Array(this.dim).fill(0));
      for (let i = 0; i < hLayer.length; i++) {
        for (let j = 0; j < this.dim; j++) {
          routed[i][j] = hLayer[i][j] + attnOut[i][j];
        }
      }
      routed = clipState(routed, this.semanticIndices);

      const routedNorm = layerNorm(routed, this.semanticIndices);
      const { nextH: sergOut, activeRules } = this.executeSergLayer(routedNorm, layerIndex);

      let nextState: Matrix = Array.from({ length: hLayer.length }, () => new Array(this.dim).fill(0));
      for (let i = 0; i < hLayer.length; i++) {
        for (let j = 0; j < this.dim; j++) {
          nextState[i][j] = routed[i][j] + (sergOut[i][j] - routedNorm[i][j]);
        }
      }
      nextState = clipState(nextState, this.semanticIndices);

      let conflictVal = 0.0;
      if (nextState.length > 0) {
        let maxConflict = -Infinity;
        for (let i = 0; i < nextState.length; i++) {
          const val = nextState[i][this.conflictOffset];
          if (val > maxConflict) {
            maxConflict = val;
          }
        }
        conflictVal = maxConflict;
      }

      history.push({
        iteration,
        conflict_value: conflictVal,
        state: nextState.map(row => [...row]),
        active_rules: activeRules
      });

      hLayer = nextState.map(row => [...row]);

      if (iteration > 0 && conflictVal < epsilon) {
        converged = true;
        break;
      }

      iteration++;
    }

    return {
      hLayer,
      cceInfo: {
        converged,
        iterations: iteration + (converged ? 1 : 0),
        history
      }
    };
  }

  runForward(
    tokens: (string | number)[],
    maxLayer?: number,
    cceLayers?: number[],
    maxIter?: number,
    epsilon?: number
  ): { finalState: Matrix; trace: ExecutionTraceStep[] } {
    const actualMaxLayer = maxLayer ?? this.architecture["max_layers"] ?? 40;
    const actualCceLayers = cceLayers ?? this.architecture["cce_layers"] ?? [30, 31, 32, 33, 34];
    const actualMaxIter = maxIter ?? this.architecture["default_cce_max_iter"] ?? 10;
    const actualEpsilon = epsilon ?? this.architecture["default_cce_epsilon"] ?? 0.1;
    const states: Matrix = [];
    for (const t of tokens) {
      if (typeof t === "string") {
        states.push(this.getTokenVector(t));
      } else {
        states.push([...this.embedding[t]]);
      }
    }

    let H = states;
    const posOffset = this.tsrMap["SYNTAX::POSITION_INDEX"] ? this.tsrMap["SYNTAX::POSITION_INDEX"][0] : -1;
    if (posOffset !== -1) {
      for (let idx = 0; idx < H.length; idx++) {
        H[idx][posOffset] = idx;
      }
    }
    const trace: ExecutionTraceStep[] = [];

    for (let l = 0; l < actualMaxLayer; l++) {
      if (actualCceLayers.includes(l)) {
        const { hLayer, cceInfo } = this.executeCceLayer(H, l, actualMaxIter, actualEpsilon);
        H = hLayer;
        trace.push({
          layer: l,
          type: "CCE",
          cce_info: cceInfo
        });
      } else {
        if (!(l in this.serg)) {
          trace.push({
            layer: l,
            type: "STANDARD",
            active_rules: Array.from({ length: H.length }, () => [])
          });
          continue;
        }

        const hNorm = layerNorm(H, this.semanticIndices);
        const attnOut = this.executeAttention(hNorm);
        
        let routed: Matrix = Array.from({ length: H.length }, () => new Array(this.dim).fill(0));
        for (let i = 0; i < H.length; i++) {
          for (let j = 0; j < this.dim; j++) {
            routed[i][j] = H[i][j] + attnOut[i][j];
          }
        }
        routed = clipState(routed, this.semanticIndices);

        const routedNorm = layerNorm(routed, this.semanticIndices);
        const { nextH: sergOut, activeRules } = this.executeSergLayer(routedNorm, l);

        const nextState: Matrix = Array.from({ length: H.length }, () => new Array(this.dim).fill(0));
        for (let i = 0; i < H.length; i++) {
          for (let j = 0; j < this.dim; j++) {
            nextState[i][j] = routed[i][j] + (sergOut[i][j] - routedNorm[i][j]);
          }
        }
        H = clipState(nextState, this.semanticIndices);

        trace.push({
          layer: l,
          type: "STANDARD",
          active_rules: activeRules
        });
      }
    }

    return { finalState: H, trace };
  }
}
