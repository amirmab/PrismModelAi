import { loadEagerManifest } from './src/cnvm/loader';
import { CNVMCompiler } from './src/cnvm/compiler';
import { CNVMRuntime, layerNorm, clipState } from './src/cnvm/runtime';

const manifest = loadEagerManifest();
const compiler = new CNVMCompiler(manifest.vocabData, manifest.routingData, manifest.rulesData, manifest.tsrMap, manifest.dim);
const compiled = compiler.compile();
const runtime = new CNVMRuntime(compiled, manifest.tsrMap, manifest.dim);

let H = [runtime.getTokenVector("canada")];
let l = 1;
let H_norm = layerNorm(H, runtime.semanticIndices);
let attn = runtime.executeAttention(H_norm);
let routed = clipState(H.map((row, i) => row.map((val, j) => val + attn[i][j])), runtime.semanticIndices);
let routed_norm = layerNorm(routed, runtime.semanticIndices);
let { nextH: sergOut } = runtime.executeSergLayer(routed_norm, l);
let nextState = routed.map((row, i) => row.map((val, j) => val + (sergOut[i][j] - routed_norm[i][j])));
H = clipState(nextState, runtime.semanticIndices);

l = 2;
H_norm = layerNorm(H, runtime.semanticIndices);
attn = runtime.executeAttention(H_norm);
routed = clipState(H.map((row, i) => row.map((val, j) => val + attn[i][j])), runtime.semanticIndices);
routed_norm = layerNorm(routed, runtime.semanticIndices);
let sergOut2 = runtime.executeSergLayer(routed_norm, l).nextH;
nextState = routed.map((row, i) => row.map((val, j) => val + (sergOut2[i][j] - routed_norm[i][j])));
H = clipState(nextState, runtime.semanticIndices);

console.log("TS Layer 2 Output (SYNTAX::SUBJECT index 13):", H[0][13]);
console.log("TS Layer 2 Output (SYNTAX::NOUN index 11):", H[0][11]);

l = 3;
H_norm = layerNorm(H, runtime.semanticIndices);
attn = runtime.executeAttention(H_norm);
routed = clipState(H.map((row, i) => row.map((val, j) => val + attn[i][j])), runtime.semanticIndices);
console.log("TS Layer 3 Routed (SYNTAX::SUBJECT index 13):", routed[0][13]);
let rn = layerNorm(routed, runtime.semanticIndices);
console.log("TS Layer 3 Routed Norm (SYNTAX::SUBJECT index 13):", rn[0][13]);
let sergOut3 = runtime.executeSergLayer(rn, l).nextH;
nextState = routed.map((row, i) => row.map((val, j) => val + (sergOut3[i][j] - rn[i][j])));
H = clipState(nextState, runtime.semanticIndices);
console.log("TS Layer 3 Output (SYS::CONFIDENCE index 6):", H[0][6]);
