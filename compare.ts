import { loadEagerManifest } from './dashboard/src/cnvm/loader';
import { CNVMCompiler } from './dashboard/src/cnvm/compiler';
import { CNVMRuntime } from './dashboard/src/cnvm/runtime';
import * as fs from 'fs';

const manifest = loadEagerManifest();
const compiler = new CNVMCompiler(manifest.vocabData, manifest.routingData, manifest.rulesData, manifest.tsrMap, manifest.dim);
const checkpoint = compiler.compile();
const runtime = new CNVMRuntime(checkpoint, manifest.tsrMap, manifest.dim);

const res = runtime.runForward(["canada", "capital"], 40, [30, 31, 32, 33, 34], 10, 0.1);
const finalState = res.finalState[res.finalState.length - 1];

const mapped: any = {};
for (const [name, [start, end]] of Object.entries(manifest.tsrMap)) {
    mapped[name] = finalState[start];
}

console.log(JSON.stringify(mapped, null, 2));
