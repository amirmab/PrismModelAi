import { describe, it, expect, beforeAll } from 'vitest';
import { loadEagerManifest } from './loader';
import type { ManifestData } from './loader';
import { CNVMCompiler } from './compiler';
import { CNVMRuntime } from './runtime';
import type { CompiledCheckpoint } from './compiler';
import type { ExecutionTraceStep } from './runtime';

describe('CNVMRuntime mathematical verification', () => {
  let manifest: ManifestData;
  let checkpoint: CompiledCheckpoint;
  let runtime: CNVMRuntime;

  beforeAll(() => {
    manifest = loadEagerManifest();
    const compiler = new CNVMCompiler(
      manifest.vocabData,
      manifest.routingData,
      manifest.rulesData,
      manifest.tsrMap,
      manifest.dim
    );
    checkpoint = compiler.compile();
    runtime = new CNVMRuntime(checkpoint, manifest.tsrMap, manifest.dim);
    console.log("TS manifest dim:", manifest.dim);
  });

  const examples = [
    // --- Core examples (short) ---
    [["canada", "capital"], "ottawa"],
    [["first", "prime_minister", "canada"], "john_a_macdonald"],
    [["year", "confederation"], "1867"],
    [["national", "symbol", "canada"], "maple_leaf"],
    [["bake", "temperature", "souffle"], "375f"],
    [["bread", "rise"], "yeast"],
    [["steak", "medium_rare"], "140f"],
    [["safest", "cloth"], "microfiber"],
    [["protects", "uv"], "carnauba_wax"],
    [["cleans", "leather", "seats"], "ph_neutral_cleaner"],
    // --- Permutation tests ---
    [["capital", "canada"], "ottawa"],
    [["first", "canada", "prime_minister"], "john_a_macdonald"],
    [["medium_rare", "steak"], "140f"],
    [["rise", "bread"], "yeast"],
    [["souffle", "bake", "temperature"], "375f"],
  ];

  const longer_examples = [
    [["steak", "temperature", "medium_rare"], "140f"],
    [["souffle", "bake", "temperature", "rise"], "375f"],
    [["bread", "bake", "rise"], "yeast"],
    [["steak", "bake", "medium_rare"], "140f"],
    [["first", "prime_minister", "canada", "year"], "john_a_macdonald"],
    [["canada", "national", "symbol"], "maple_leaf"],
    [["capital", "canada", "first"], "ottawa"],
    [["cleans", "protects", "leather"], "ph_neutral_cleaner"],
  ];

  const computeSimilarities = (finalState: number[]) => {
    const similarities: Record<string, number> = {};
    for (const [tokenName, rule] of Object.entries(manifest.outputData)) {
      let score = 0;
      let count = 0;
      const targetSliders = (rule as any).target_sliders || {};
      for (const [sliderName, sliderConfig] of Object.entries(targetSliders)) {
        const targetWeight = (sliderConfig as any).weight;
        const startOffset = manifest.tsrMap[sliderName]?.[0];
        if (startOffset !== undefined) {
          const actualVal = finalState[startOffset];
          const diff = actualVal - targetWeight;
          score += diff * diff;
          count++;
        }
      }
      const mse = count > 0 ? score / count : Infinity;
      similarities[tokenName] = Math.max(0, Math.min(100, Math.round((1.0 - mse / 8.0) * 100)));
    }
    return similarities;
  };

  it('passes Tier 1 Canonical projection strictly', () => {


    for (const [tokens, expected_output] of examples) {
      const res = runtime.runForward(tokens as string[], 40, [30, 31, 32, 33, 34], 10, 0.1);
      let finalState = new Array(res.finalState[0].length).fill(0);
      for (let i = 0; i < res.finalState.length; i++) {
        for (let j = 0; j < finalState.length; j++) {
          finalState[j] += res.finalState[i][j];
        }
      }
      for (let j = 0; j < finalState.length; j++) {
        finalState[j] /= res.finalState.length;
      }
      const similarities = computeSimilarities(finalState);
      
      const expectedSimilarity = similarities[expected_output as string] ?? -1;
      if (expectedSimilarity < 90) {
        console.log(`Failed canonical ${tokens}. Similarities:`, similarities);
        console.log(`FINAL VECTOR FOR ${tokens}:`, JSON.stringify(finalState));
      }
      expect(expectedSimilarity).toBeGreaterThanOrEqual(90);

      for (const [tokenName, sim] of Object.entries(similarities)) {
        if (tokenName !== expected_output) {
          if (sim > expectedSimilarity) {
             throw new Error(`[Canonical] Strict isolation failed on ${tokens}. ${tokenName} had ${sim}% which beat ${expected_output} (${expectedSimilarity}%)`);
          }
        }
      }
    }
  });

  it('passes Tier 2 Longer projection Rank-1', () => {
    for (const [tokens, expected_output] of longer_examples) {
      const res = runtime.runForward(tokens as string[], 40, [30, 31, 32, 33, 34], 10, 0.1);
      let finalState = new Array(res.finalState[0].length).fill(0);
      for (let i = 0; i < res.finalState.length; i++) {
        for (let j = 0; j < finalState.length; j++) {
          finalState[j] += res.finalState[i][j];
        }
      }
      for (let j = 0; j < finalState.length; j++) {
        finalState[j] /= res.finalState.length;
      }
      const similarities = computeSimilarities(finalState);
      
      const maxScore = Math.max(...Object.values(similarities));
      const expectedScore = similarities[expected_output as string] ?? -1;
      
      if (expectedScore !== maxScore) {
        const sorted = Object.entries(similarities).sort((a, b) => b[1] - a[1]).slice(0, 3);
        throw new Error(`[Longer] Rank-1 failed on ${tokens}. Expected ${expected_output} to tie for first (score ${maxScore}), but got ${expectedScore} (scores: ${JSON.stringify(sorted)})`);
      }
    }
  });

  it('verifies Grammar Parsing properties', () => {
    const resQ = runtime.runForward(["what"], 40, [30, 31, 32, 33, 34], 10, 0.1);
    const traceQ = resQ.trace;
    const findRuleQ = (id: number) => traceQ.some(t => t.type === 'STANDARD' && t.active_rules && t.active_rules[0].some(r => r.rule_id === id));
    expect(findRuleQ(101)).toBe(true);
    expect(findRuleQ(107)).toBe(true);
    expect(findRuleQ(109)).toBe(true);

    const resN = runtime.runForward(["canada"], 40, [30, 31, 32, 33, 34], 10, 0.1);
    const traceN = resN.trace;
    const firedN = [];
    for (const t of traceN) {
      if (t.type === 'STANDARD' && t.active_rules && t.active_rules[0]) {
        for (const r of t.active_rules[0]) {
          firedN.push([t.layer, r.rule_id]);
        }
      }
    }
    console.log("TS Fired N:", JSON.stringify(firedN));
    
    const findRuleN = (id: number) => traceN.some(t => t.type === 'STANDARD' && t.active_rules && t.active_rules[0].some(r => r.rule_id === id));
    expect(findRuleN(102)).toBe(true);
    expect(findRuleN(104)).toBe(true);
    expect(findRuleN(106)).toBe(true);
    expect(findRuleN(110)).toBe(true);

    const resV = runtime.runForward(["bake"], 40, [30, 31, 32, 33, 34], 10, 0.1);
    const traceV = resV.trace;
    const findRuleV = (id: number) => traceV.some(t => t.type === 'STANDARD' && t.active_rules && t.active_rules[0].some(r => r.rule_id === id));
    expect(findRuleV(103)).toBe(true);
    expect(findRuleV(108)).toBe(true);
  });

  it('verifies dynamic position encoding injection', () => {
    const tokens = ["first", "prime_minister", "canada"];
    const res = runtime.runForward(tokens, 0);
    
    const posOffset = manifest.tsrMap["SYNTAX::POSITION_INDEX"]?.[0];
    expect(posOffset).toBeDefined();
    
    for (let idx = 0; idx < tokens.length; idx++) {
      expect(res.finalState[idx][posOffset!]).toBe(idx);
    }
  });

  it('verifies order sensitivity and positional rule triggers', () => {
    // 1. Forward order: "first" at index 0, "prime_minister" at index 1, "canada" at index 2
    const resA = runtime.runForward(["first", "prime_minister", "canada"], 40, [30, 31, 32, 33, 34], 10, 0.1);
    
    // 2. Reverse order: "canada" at index 0, "prime_minister" at index 1, "first" at index 2
    const resB = runtime.runForward(["canada", "prime_minister", "first"], 40, [30, 31, 32, 33, 34], 10, 0.1);

    const getFiredRule0 = (trace: typeof resA.trace) => {
      const layerTrace = trace.find(t => t.layer === 2);
      expect(layerTrace).toBeDefined();
      return layerTrace!.active_rules!.map(tokenRules => 
        tokenRules.some(r => r.rule_id === 0)
      );
    };

    const firedA = getFiredRule0(resA.trace);
    const firedB = getFiredRule0(resB.trace);

    // First element in sequence (index 0) must NOT fire rule 0 (since position value < 0.5 threshold)
    // Later elements (index 1 & 2) MUST fire rule 0 (since position values >= 1.0 & 2.0 respectively)
    expect(firedA).toEqual([false, true, true]);
    expect(firedB).toEqual([false, true, true]);

    // Compare final vectors: states must be mathematically order-dependent
    const stateA = resA.finalState;
    const stateB = resB.finalState;

    let isDifferent = false;
    for (let i = 0; i < stateA.length; i++) {
      for (let j = 0; j < stateA[i].length; j++) {
        if (stateA[i][j] !== stateB[i][j]) {
          isDifferent = true;
          break;
        }
      }
    }
    expect(isDifferent).toBe(true);
  });
});
