import { clampWeight } from "./compiler_types";
import type { CompiledCheckpoint, CompiledLayer, DomainRouting, RuleNir, VocabToken } from "./compiler_types";
import { slidersToVector } from "./tsr";

export { clampWeight };
export type { RuleNir, VocabToken, DomainRouting, CompiledLayer, CompiledCheckpoint };

export class CNVMCompiler {
  vocab_data: Record<string, VocabToken>;
  routing_data: Record<string, DomainRouting>;
  rules_data: Record<string, RuleNir>;
  tsrMap: Record<string, [number, number]>;
  dim: number;

  constructor(
    vocab_data: Record<string, VocabToken>,
    routing_data: Record<string, DomainRouting>,
    rules_data: Record<string, RuleNir>,
    tsrMap: Record<string, [number, number]>,
    dim: number
  ) {
    this.vocab_data = vocab_data;
    this.routing_data = routing_data;
    this.rules_data = rules_data;
    this.tsrMap = tsrMap;
    this.dim = dim;
  }

  private getRegisterOffset(name: string): number {
    if (!(name in this.tsrMap)) {
      throw new Error(`Unknown register name: ${name}`);
    }
    return this.tsrMap[name][0];
  }

  compileEmbedding(): { embedding: number[][]; token_to_id: Record<string, number> } {
    let maxId = 0;
    const token_to_id: Record<string, number> = {};
    
    for (const [token, data] of Object.entries(this.vocab_data)) {
      const tokenId = data.token_id;
      token_to_id[token] = tokenId;
      if (tokenId > maxId) {
        maxId = tokenId;
      }
    }

    const embedding: number[][] = Array.from(
      { length: maxId + 1 },
      () => new Array(this.dim).fill(0)
    );

    for (const [token, data] of Object.entries(this.vocab_data)) {
      const tokenId = token_to_id[token];
      const sliders: Record<string, number> = {};

      // Resolve ESC inheritance
      if (data.inherits) {
        const parentName = data.inherits;
        const parentData = this.vocab_data[parentName];
        if (parentData && parentData.sliders) {
          Object.assign(sliders, parentData.sliders);
        }
        if (data.overrides) {
          Object.assign(sliders, data.overrides);
        }
      } else {
        if (data.sliders) {
          Object.assign(sliders, data.sliders);
        }
      }

      // Clamp values and build vector
      const clampedSliders: Record<string, number> = {};
      for (const [k, v] of Object.entries(sliders)) {
        const numVal = typeof v === 'object' && v !== null && 'value' in v ? (v as any).value : v;
        clampedSliders[k] = clampWeight(Number(numVal));
      }

      embedding[tokenId] = slidersToVector(clampedSliders, this.tsrMap, this.dim);
    }

    return { embedding, token_to_id };
  }

  compileDrf(): { W_q: number[][]; W_k: number[][]; W_v: number[][] } {
    const W_q = Array.from({ length: this.dim }, () => new Array(this.dim).fill(0));
    const W_k = Array.from({ length: this.dim }, () => new Array(this.dim).fill(0));
    const W_v = Array.from({ length: this.dim }, () => new Array(this.dim).fill(0));

    // 1. Setup diagonal blocks (identity matrices) for domain isolation
    for (const [domain, [start, end]] of Object.entries(this.tsrMap)) {
      if (domain === "SYNTAX::POSITION_INDEX" || domain === "SYNTAX::POSITION_REL" || domain === "RESERVED") {
        continue;
      }
      for (let i = start; i <= end; i++) {
        W_q[i][i] = 1.0;
        W_k[i][i] = 1.0;
        if (i < 94) {
          W_v[i][i] = 1.0;
        }
      }
    }

    // 2. Punch bridges (SIDBs)
    for (const [domain, config] of Object.entries(this.routing_data)) {
      if (config.bridges) {
        try {
          const startIdx = this.getRegisterOffset(domain);
          for (const [targetDomain, weight] of Object.entries(config.bridges)) {
            const targetIdx = this.getRegisterOffset(targetDomain);
            const clampedW = clampWeight(weight);
            W_q[startIdx][targetIdx] = clampedW;
            W_k[startIdx][targetIdx] = clampedW;
            W_v[startIdx][targetIdx] = clampedW;
          }
        } catch {
          // Skip if domain doesn't exist in tsrMap to prevent crash during dynamic edits
        }
      }
    }

    // 3. Normalize Q and K matrices by baseline_d^(1/4) so logits product Q K^T is scaled by sqrt(baseline_d)
    const scalingFactor = Math.pow(94.0, 0.25);
    for (let i = 0; i < this.dim; i++) {
      for (let j = 0; j < this.dim; j++) {
        W_q[i][j] /= scalingFactor;
        W_k[i][j] /= scalingFactor;
      }
    }

    return { W_q, W_k, W_v };
  }

  compileSerg(): Record<number, CompiledLayer> {
    const rulesByLayer: Record<number, RuleNir[]> = {};
    for (const rule of Object.values(this.rules_data)) {
      const lIdx = rule.layer_index;
      if (!rulesByLayer[lIdx]) {
        rulesByLayer[lIdx] = [];
      }
      rulesByLayer[lIdx].push(rule);
    }

    const serg: Record<number, CompiledLayer> = {};

    for (const [lStr, rules] of Object.entries(rulesByLayer)) {
      const lIdx = parseInt(lStr, 10);
      const numRules = rules.length;

      const W_in = Array.from({ length: this.dim }, () => new Array(numRules).fill(0));
      const W_out = Array.from({ length: numRules }, () => new Array(this.dim).fill(0));
      const b = new Array(numRules).fill(-0.5); // Default bias is -0.5 (threshold)
      const ruleIds = rules.map(r => r.rule_id);

      for (let c = 0; c < numRules; c++) {
        const rule = rules[c];
        try {
          const triggerOffset = this.getRegisterOffset(rule.trigger_slider_name);
          const resultOffset = this.getRegisterOffset(rule.result_slider_name);
          const gateIn = clampWeight(rule.gate_in_weight);
          const gateOut = clampWeight(rule.gate_out_weight);

          // Preconditions
          W_in[triggerOffset][c] = gateIn;

          // Postconditions
          W_out[c][resultOffset] = gateOut;
          
          // Provenance Rule ID write-back to META::PROVENANCE (if it exists)
          if ("META::PROVENANCE" in this.tsrMap) {
            const provOffset = this.getRegisterOffset("META::PROVENANCE");
            W_out[c][provOffset] = rule.rule_id;
          }
        } catch {
          // Skip missing registers to handle dirty editing states gracefully
        }
      }

      serg[lIdx] = { W_in, W_out, b, rule_ids: ruleIds };
    }

    return serg;
  }

  compile(): CompiledCheckpoint {
    const { embedding, token_to_id } = this.compileEmbedding();
    const { W_q, W_k, W_v } = this.compileDrf();
    const serg = this.compileSerg();

    const id_to_token: Record<number, string> = {};
    for (const [k, v] of Object.entries(token_to_id)) {
      id_to_token[v] = k;
    }

    return {
      embedding,
      token_to_id,
      id_to_token,
      W_q,
      W_k,
      W_v,
      serg
    };
  }
}
