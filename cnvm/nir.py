"""
cnvm/nir.py — Neural Intermediate Representation loader and validator.

Reads the manifest/ directory structure and returns validated dicts
ready for the CNVMCompiler. No configuration lives in this file.

Manifest directory layout expected:
  manifest/
    sliders.json          ← register layout (loaded by tsr.py)
    vocabulary.json       ← token definitions with slider values
    output_rules.json     ← output token target conditions
    layers/
      layer_N/
        RULE_*.json       ← one FFN rule per file
        DOMAIN_A_DOMAIN_B.json  ← one attention bridge per file
"""

import json
import os
from dataclasses import dataclass, field


class ValidationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    """Loads and returns a JSON file, raising a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _is_attention_bridge(filename: str) -> bool:
    """
    Detects attention bridge files by the naming convention:
      DOMAIN_A--DOMAIN_B.json  (contains '--', no 'RULE_' prefix)
    """
    name = os.path.splitext(filename)[0]
    return "--" in name and not name.startswith("RULE_")


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_vocabulary(vocabulary_data: dict, tsr_map: dict) -> None:
    for token, info in vocabulary_data.items():
        if "token_id" not in info:
            raise ValidationError(f"Token '{token}' is missing 'token_id'.")

        if "inherits" in info:
            parent = info["inherits"]
            if parent not in vocabulary_data:
                raise ValidationError(
                    f"Token '{token}' inherits from unknown token '{parent}'."
                )

        for section in ("sliders", "overrides"):
            for slider_name in info.get(section, {}):
                if slider_name not in tsr_map:
                    raise ValidationError(
                        f"Token '{token}' references unknown register '{slider_name}' in '{section}'."
                    )


def validate_routing(routing_data: dict, tsr_map: dict) -> None:
    for domain, config in routing_data.items():
        if domain not in tsr_map:
            raise ValidationError(
                f"Routing config references unknown domain: '{domain}'"
            )
        for target_domain, weight in config.get("bridges", {}).items():
            if target_domain not in tsr_map:
                raise ValidationError(
                    f"Bridge in '{domain}' points to unknown domain: '{target_domain}'"
                )
            try:
                float(weight)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"Bridge weight between '{domain}' and '{target_domain}' must be a float."
                )


def validate_rules(rules_data: dict, tsr_map: dict) -> None:
    required_keys = [
        "rule_id", "trigger_slider_name", "result_slider_name",
        "gate_in_weight", "gate_out_weight"
    ]
    for rule_name, rule_info in rules_data.items():
        for key in required_keys:
            if key not in rule_info:
                raise ValidationError(
                    f"Rule '{rule_name}' is missing required key '{key}'."
                )

        for slider_key in ("trigger_slider_name", "result_slider_name"):
            slider = rule_info[slider_key]
            if slider not in tsr_map:
                raise ValidationError(
                    f"Rule '{rule_name}' references unknown register '{slider}' in '{slider_key}'."
                )

        try:
            int(rule_info["rule_id"])
            float(rule_info["gate_in_weight"])
            float(rule_info["gate_out_weight"])
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Rule '{rule_name}' has invalid numeric value: {e}"
            )


# ---------------------------------------------------------------------------
# Manifest Loader
# ---------------------------------------------------------------------------

@dataclass
class ManifestData:
    """Validated, structured data loaded from the manifest directory."""
    vocab_data: dict = field(default_factory=dict)
    routing_data: dict = field(default_factory=dict)   # domain -> {bridges: {target: weight}}
    rules_data: dict = field(default_factory=dict)      # rule_name -> rule dict (with layer_index)
    output_data: dict = field(default_factory=dict)
    tsr_map: dict = field(default_factory=dict)


def load_manifest(manifest_dir: str = "manifest") -> ManifestData:
    """
    Walks the manifest directory and loads all JSON configuration.

    Separation logic for layer files:
      - Files starting with RULE_ → FFN rules
      - Files with pattern DOMAIN_A_DOMAIN_B → attention bridges

    Returns a validated ManifestData instance.
    """
    # Step 1: Load the slider map first (needed for validation)
    from cnvm.tsr import load_slider_map
    tsr_map = load_slider_map(manifest_dir)

    # Step 2: Vocabulary
    vocab_data = _load_json(os.path.join(manifest_dir, "vocabulary.json"))

    # Normalise vocabulary: flatten slider dicts {name: {value, description}} → {name: value}
    # while preserving inherits/overrides structure
    flat_vocab: dict = {}
    for token, info in vocab_data.items():
        entry: dict = {"token_id": info["token_id"]}
        if "concept_description" in info:
            entry["concept_description"] = info["concept_description"]
        if "inherits" in info:
            entry["inherits"] = info["inherits"]
        if "sliders" in info:
            entry["sliders"] = {
                k: v["value"] if isinstance(v, dict) else v
                for k, v in info["sliders"].items()
            }
        if "overrides" in info:
            entry["overrides"] = {
                k: v["value"] if isinstance(v, dict) else v
                for k, v in info["overrides"].items()
            }
        flat_vocab[token] = entry

    validate_vocabulary(flat_vocab, tsr_map)

    # Step 3: Walk layers/ directory for rules and attention bridges
    rules_data: dict = {}
    routing_data: dict = {}   # built from per-layer bridge files

    layers_dir = os.path.join(manifest_dir, "layers")
    if os.path.isdir(layers_dir):
        for layer_dir_name in sorted(os.listdir(layers_dir)):
            layer_path = os.path.join(layers_dir, layer_dir_name)
            if not os.path.isdir(layer_path):
                continue

            # Parse layer index from directory name "layer_N"
            try:
                layer_index = int(layer_dir_name.split("_")[1])
            except (IndexError, ValueError):
                raise ValidationError(
                    f"Layer directory '{layer_dir_name}' must be named 'layer_N' "
                    f"where N is an integer."
                )

            for filename in sorted(os.listdir(layer_path)):
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(layer_path, filename)
                data = _load_json(filepath)

                if _is_attention_bridge(filename):
                    # Attention bridge: parse DOMAIN_A--DOMAIN_B.json
                    stem = os.path.splitext(filename)[0]
                    separator_idx = stem.find("--")
                    if separator_idx == -1:
                        raise ValidationError(
                            f"Bridge file '{filename}' must use '--' to separate two domain names."
                        )
                    source_domain = stem[:separator_idx]
                    target_domain = stem[separator_idx + 2:]
                    if source_domain not in tsr_map:
                        raise ValidationError(
                            f"Bridge file '{filename}': source domain '{source_domain}' not in tsr_map."
                        )
                    if target_domain not in tsr_map:
                        raise ValidationError(
                            f"Bridge file '{filename}': target domain '{target_domain}' not in tsr_map."
                        )

                    if source_domain not in routing_data:
                        routing_data[source_domain] = {"bridges": {}}
                    # Use the average of q and k weights as the routing strength
                    bridge_weight = (
                        float(data["q"]["value"]) + float(data["k"]["value"])
                    ) / 2.0
                    routing_data[source_domain]["bridges"][target_domain] = bridge_weight

                else:
                    # FFN rule file: inject layer_index from directory name
                    rule_name = os.path.splitext(filename)[0]
                    rule_entry = dict(data)
                    rule_entry["layer_index"] = layer_index
                    rules_data[rule_name] = rule_entry

    validate_routing(routing_data, tsr_map)
    validate_rules(rules_data, tsr_map)

    # Step 4: Output rules (optional)
    output_path = os.path.join(manifest_dir, "output_rules.json")
    output_data = _load_json(output_path) if os.path.exists(output_path) else {}

    return ManifestData(
        vocab_data=flat_vocab,
        routing_data=routing_data,
        rules_data=rules_data,
        output_data=output_data,
        tsr_map=tsr_map,
    )
