import { loadSliderMap } from "./tsr";
import type { SlidersConfig } from "./tsr";
import type { VocabToken, DomainRouting, RuleNir } from "./compiler_types";

// Eagerly import all JSON configurations from the manifest folder outside dashboard
const jsonModules = import.meta.glob("../../../manifest/**/*.json", { eager: true });

export interface ManifestData {
  vocabData: Record<string, VocabToken>;
  routingData: Record<string, DomainRouting>;
  rulesData: Record<string, RuleNir>;
  outputData: Record<string, any>;
  slidersConfig: SlidersConfig;
  tsrMap: Record<string, [number, number]>;
  dim: number;
  architecture: any;
}

export function getEagerManifestModules(): Record<string, any> {
  return jsonModules;
}

export function parseManifest(modules: Record<string, any>): ManifestData {
  let slidersConfig: SlidersConfig = {};
  let rawVocab: Record<string, any> = {};
  let rawOutput: Record<string, any> = {};
  let rawArchitecture: any = {
    max_layers: 40,
    cce_layers: [30, 31, 32, 33, 34],
    default_cce_max_iter: 10,
    default_cce_epsilon: 0.1,
    layers: []
  };
  
  const rulesData: Record<string, RuleNir> = {};
  const routingData: Record<string, DomainRouting> = {};
  const rawBridges: { fileName: string; data: any }[] = [];

  for (const [path, module] of Object.entries(modules)) {
    const data = module && (module.default !== undefined ? module.default : module);
    if (!data) continue;

    if (path.endsWith("sliders.json")) {
      slidersConfig = data;
    } else if (path.endsWith("vocabulary.json")) {
      rawVocab = data;
    } else if (path.endsWith("output_rules.json")) {
      rawOutput = data;
    } else if (path.endsWith("architecture.json")) {
      rawArchitecture = data;
    } else if (path.includes("/layers/")) {
      const parts = path.split("/");
      const fileName = parts[parts.length - 1];
      const layerDir = parts[parts.length - 2];
      
      const layerIndex = parseInt(layerDir.split("_")[1], 10);
      if (isNaN(layerIndex)) continue;

      const ruleName = fileName.replace(".json", "");

      if (fileName.includes("--") && !fileName.startsWith("RULE_")) {
        rawBridges.push({ fileName, data });
      } else {
        rulesData[ruleName] = {
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

  // Calculate tsrMap & dim
  const { tsrMap, dim } = loadSliderMap(slidersConfig);

  // Now process bridges using tsrMap
  for (const { fileName, data } of rawBridges) {
    const stem = fileName.replace(".json", "");

    // Bridge filenames use '--' as the unambiguous separator between two domain names.
    // e.g. "DOMAIN::CAR_CLEANING--SYS::INTEGRITY.json"
    const separatorIdx = stem.indexOf("--");
    if (separatorIdx === -1) continue;

    const sourceDomain = stem.slice(0, separatorIdx);
    const targetDomain = stem.slice(separatorIdx + 2);

    if (sourceDomain in tsrMap && targetDomain in tsrMap) {
      if (!routingData[sourceDomain]) {
        routingData[sourceDomain] = { bridges: {} };
      }
      const qVal = data.q && data.q.value !== undefined ? Number(data.q.value) : 0;
      const kVal = data.k && data.k.value !== undefined ? Number(data.k.value) : 0;
      const weight = (qVal + kVal) / 2.0;

      if (!routingData[sourceDomain].bridges) {
        routingData[sourceDomain].bridges = {};
      }
      routingData[sourceDomain].bridges![targetDomain] = weight;
    }
  }

  // Flatten vocab
  const vocabData: Record<string, VocabToken> = {};
  for (const [token, info] of Object.entries(rawVocab)) {
    const entry: VocabToken = {
      token_id: info.token_id,
      concept_description: info.concept_description,
      inherits: info.inherits
    };
    
    if (info.sliders) {
      entry.sliders = {};
      for (const [k, v] of Object.entries(info.sliders)) {
        entry.sliders[k] = typeof v === "object" && v !== null && "value" in v ? (v as any).value : Number(v);
      }
    }

    if (info.overrides) {
      entry.overrides = {};
      for (const [k, v] of Object.entries(info.overrides)) {
        entry.overrides[k] = typeof v === "object" && v !== null && "value" in v ? (v as any).value : Number(v);
      }
    }
    vocabData[token] = entry;
  }

  return {
    vocabData,
    routingData,
    rulesData,
    outputData: rawOutput,
    slidersConfig,
    tsrMap,
    dim,
    architecture: rawArchitecture,
  };
}

export function loadEagerManifest(): ManifestData {
  return parseManifest(jsonModules);
}
