import numpy as np
from cnvm.tsr import get_dim, get_tsr_map, get_register_offset, sliders_to_vector


def clamp_weight(val: float, limit: float = 2.0) -> float:
    """Clamps a weight value to [-limit, limit] to prevent activation explosions."""
    return max(min(float(val), limit), -limit)


class CNVMCompiler:
    def __init__(self, vocab_data: dict, routing_data: dict, rules_data: dict, architecture_data: dict = None):
        self.vocab_data = vocab_data
        self.routing_data = routing_data
        self.rules_data = rules_data
        self.architecture_data = architecture_data or {
            "max_layers": 40,
            "cce_layers": [30, 31, 32, 33, 34],
            "default_cce_max_iter": 10,
            "default_cce_epsilon": 0.1
        }

    def compile_embedding(self) -> tuple[np.ndarray, dict]:
        """
        Compiles the vocabulary ontology with ESC inheritance handling.

        Returns:
            embedding_matrix: np.ndarray of shape (max_token_id + 1, DIM)
            token_to_id: dict mapping token string → integer ID
        """
        DIM = get_dim()
        max_id = 0
        token_to_id: dict[str, int] = {}

        for token, data in self.vocab_data.items():
            token_id = int(data["token_id"])
            token_to_id[token] = token_id
            if token_id > max_id:
                max_id = token_id

        # Zero-initialise the full embedding matrix
        embedding_matrix = np.zeros((max_id + 1, DIM), dtype=np.float32)
        
        # Set default value of safety facts to -2.0
        safety_facts = {
            "FACT::BOILER_ACTIVE",
            "FACT::WATER_LEVEL_LOW",
            "FACT::STEAM_PRESSURE_HIGH",
            "FACT::TEMPERATURE_SCALD",
            "FACT::SAFETY_VALVE_STUCK"
        }
        tsr_map = get_tsr_map()
        for name in safety_facts:
            if name in tsr_map:
                idx = tsr_map[name][0]
                embedding_matrix[:, idx] = -2.0

        system_defaults = {
            "SYS::CONFIDENCE": 2.0,
            "SYS::INTEGRITY": 2.0,
        }
        for name, val in system_defaults.items():
            if name in tsr_map:
                idx = tsr_map[name][0]
                embedding_matrix[:, idx] = val

        for token, data in self.vocab_data.items():
            token_id = token_to_id[token]
            sliders: dict[str, float] = {}

            # Resolve ESC Inheritance
            if "inherits" in data:
                parent_name = data["inherits"]
                parent_data = self.vocab_data.get(parent_name, {})
                if "sliders" in parent_data:
                    sliders.update(parent_data["sliders"])
                if "overrides" in data:
                    sliders.update(data["overrides"])
            else:
                if "sliders" in data:
                    sliders.update(data["sliders"])

            # Clamp all values before writing
            clamped = {k: clamp_weight(v) for k, v in sliders.items()}
            
            token_vector = embedding_matrix[token_id].copy()
            for name, value in clamped.items():
                if name in tsr_map:
                    idx = tsr_map[name][0]
                    token_vector[idx] = float(value)
            embedding_matrix[token_id] = token_vector

        return embedding_matrix, token_to_id

    def compile_drf(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compiles block-diagonal Domain Routing Fabric matrices (W_q, W_k, W_v).

        - Enforces block-diagonal identity for domain isolation.
        - Inserts Sparse Inter-Domain Bridges (SIDBs) from routing_data.
        - Normalises Q and K by d^(1/4) so Q K^T is pre-scaled by sqrt(d).

        Returns:
            W_q, W_k, W_v — each np.ndarray of shape (DIM, DIM)
        """
        DIM = get_dim()
        tsr_map = get_tsr_map()

        W_q = np.zeros((DIM, DIM), dtype=np.float32)
        W_k = np.zeros((DIM, DIM), dtype=np.float32)
        W_v = np.zeros((DIM, DIM), dtype=np.float32)

        # 1. Identity blocks for intra-domain routing isolation
        for domain, (start, end) in tsr_map.items():
            if domain in {"SYNTAX::POSITION_INDEX", "SYNTAX::POSITION_REL", "RESERVED", "META::PROVENANCE", "META::EVIDENCE", "SYS::CONFLICT"}:
                continue
            for i in range(start, end + 1):
                if i < 94 or i == 106:
                    W_q[i, i] = 1.0
                    W_k[i, i] = 1.0
                    W_v[i, i] = 1.0

        # 2. Compile Sparse Inter-Domain Bridges (SIDBs)
        for domain, config in self.routing_data.items():
            if "bridges" not in config:
                continue
            start_idx = get_register_offset(domain)
            for target_domain, weight in config["bridges"].items():
                target_idx = get_register_offset(target_domain)
                clamped_w = clamp_weight(weight)
                W_q[start_idx, target_idx] = clamped_w
                W_k[start_idx, target_idx] = clamped_w
                W_v[start_idx, target_idx] = clamped_w

        # 3. Normalise: divide Q and K by baseline_d^0.25 so Q K^T / sqrt(baseline_d) is stable
        scaling_factor = 94.0 ** 0.25
        W_q_norm = W_q / scaling_factor
        W_k_norm = W_k / scaling_factor

        return W_q_norm, W_k_norm, W_v

    def compile_serg(self) -> dict[int, dict]:
        """
        Compiles rules_data into Sparse Executable Rule Graph matrices, grouped by layer.

        Each entry in the returned dict maps layer_index → {W_in, W_out, b, rule_ids}.
          W_in:  (DIM, R)  — precondition key detectors
          W_out: (R, DIM)  — postcondition effect + provenance writers
          b:     (R,)      — threshold bias (default -0.5)
        """
        DIM = get_dim()

        # Group rules by layer_index
        rules_by_layer: dict[int, list] = {}
        for rule_name, rule_info in self.rules_data.items():
            l_idx = int(rule_info["layer_index"])
            rules_by_layer.setdefault(l_idx, []).append(rule_info)

        compiled_layers: dict[int, dict] = {}
        for l_idx, rules in rules_by_layer.items():
            R = len(rules)
            W_in  = np.zeros((DIM, R), dtype=np.float32)
            W_out = np.zeros((R, DIM), dtype=np.float32)
            b     = np.full(R, -0.5, dtype=np.float32)   # threshold bias per paper §8.1

            for c, rule in enumerate(rules):
                trigger_offset = get_register_offset(rule["trigger_slider_name"])
                result_offset  = get_register_offset(rule["result_slider_name"])
                gate_in  = clamp_weight(rule["gate_in_weight"])
                gate_out = clamp_weight(rule["gate_out_weight"])
                rule_id  = int(rule["rule_id"])

                # Precondition key detector
                W_in[trigger_offset, c] = gate_in

                # Postcondition: effect on result register
                W_out[c, result_offset] = gate_out

                # Provenance: write rule_id into PROVENANCE_ID register
                prov_offset = get_register_offset("META::PROVENANCE")
                W_out[c, prov_offset] = 0.0

            compiled_layers[l_idx] = {
                "W_in": W_in,
                "W_out": W_out,
                "b": b,
                "rule_ids": [r["rule_id"] for r in rules],
            }

        return compiled_layers

    def compile(self) -> dict:
        """Compiles the full CNVM tensor checkpoint from all NIR namespaces."""
        embedding_matrix, token_to_id = self.compile_embedding()
        W_q, W_k, W_v = self.compile_drf()
        serg_layers = self.compile_serg()

        return {
            "embedding":   embedding_matrix,
            "token_to_id": token_to_id,
            "id_to_token": {v: k for k, v in token_to_id.items()},
            "W_q":  W_q,
            "W_k":  W_k,
            "W_v":  W_v,
            "serg": serg_layers,
            "architecture": self.architecture_data,
        }
