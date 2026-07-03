import { loadEagerManifest } from './src/cnvm/loader';
import { CNVMCompiler } from './src/cnvm/compiler';
import { CNVMRuntime } from './src/cnvm/runtime';
const manifest = loadEagerManifest();
const compiler = new CNVMCompiler(manifest.vocabData, manifest.routingData, manifest.rulesData, manifest.tsrMap, manifest.dim);
const compiled = compiler.compile();
const runtime = new CNVMRuntime(compiled, manifest.tsrMap, manifest.dim);
const res = runtime.runForward(["canada"], 40, [30,31,32,33,34], 10, 0.1);
const fired = [];
for (const t of res.trace) {
    if (t.type === 'STANDARD') {
        if (t.active_rules && t.active_rules[0]) {
            for (const r of t.active_rules[0]) {
                fired.push([t.layer, r.rule_id]);
            }
        }
    }
}
console.log("Fired rules in TS:", JSON.stringify(fired));
