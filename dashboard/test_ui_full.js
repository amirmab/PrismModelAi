import fs from 'fs';
import path from 'path';

// Load modules manually
const getJson = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));

const vocabData = getJson('../manifest/vocabulary.json');
const routingData = getJson('../manifest/domain_routing.json');
const outputData = getJson('../manifest/output_rules.json');
const slidersConfig = getJson('../manifest/sliders.json');

const rulesData = {};
function readRules(dir) {
    if (!fs.existsSync(dir)) return;
    const files = fs.readdirSync(dir);
    for (const f of files) {
        const full = path.join(dir, f);
        if (fs.statSync(full).isDirectory()) {
            readRules(full);
        } else if (f.endsWith('.json')) {
            const data = getJson(full);
            if (f.startsWith('RULE_')) {
                const layerStr = path.basename(dir);
                const layerIndex = parseInt(layerStr.split('_')[1], 10);
                rulesData[f.replace('.json', '')] = {
                    rule_id: data.rule_id,
                    layer_index: layerIndex,
                    trigger_slider_name: data.trigger_slider_name,
                    result_slider_name: data.result_slider_name,
                    gate_in_weight: data.gate_in_weight,
                    gate_out_weight: data.gate_out_weight,
                    intent_description: data.intent_description
                };
            }
        }
    }
}
readRules('../manifest/layers');

let maxBlock = 10;
for (const entry of Object.values(slidersConfig)) {
    const size = entry.block_size !== undefined ? entry.block_size : 10;
    if (size > maxBlock) maxBlock = size;
}
const stride = Math.max(maxBlock, 10);

const tsrMap = {};
let maxCoord = 0;
for (const [name, entry] of Object.entries(slidersConfig)) {
    const coord = entry.coordinate;
    if (coord > maxCoord) maxCoord = coord;
    const blockSize = entry.block_size !== undefined ? entry.block_size : stride;
    const start = coord * stride;
    const end = start + blockSize - 1;
    tsrMap[name] = [start, end];
}
const dim = (maxCoord + 1) * stride;

console.log("TS DIM:", dim);
console.log("HISTORY_CANADA mapped to:", tsrMap["HISTORY_CANADA"]);
