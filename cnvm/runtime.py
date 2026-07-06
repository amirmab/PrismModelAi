import numpy as np
from cnvm.tsr import get_dim, get_tsr_map, get_register_offset, format_state


def _get_semantic_indices():
    """
    Dynamically computes which register indices are 'semantic' (i.e., should be
    included in layer normalization and value clipping).

    Excludes META::PROVENANCE and META::EVIDENCE metadata registers.
    This mirrors getSemanticIndices() in the TypeScript runtime.
    """
    tsr_map = get_tsr_map()
    dim = 94  # Lock to baseline semantic dimension

    metadata_names = {
        "META::PROVENANCE",
        "META::EVIDENCE",
        "RESERVED",
    }
    metadata_indices = set()
    for name in metadata_names:
        if name in tsr_map:
            start, end = tsr_map[name]
            for i in range(start, end + 1):
                if i < dim:
                    metadata_indices.add(i)

    return sorted(i for i in range(dim) if i not in metadata_indices)


def layer_norm(x, eps=1e-5):
    """Applies Layer Normalization only over the semantic domains, ignoring metadata."""
    res = x.copy()
    semantic_indices = _get_semantic_indices()
    is_flat = res.ndim == 1
    if is_flat:
        semantic_slice = res[semantic_indices]
        mean = np.mean(semantic_slice)
        var = np.var(semantic_slice)
        res[semantic_indices] = (semantic_slice - mean) / np.sqrt(var + eps)
    else:
        semantic_slice = res[:, semantic_indices]
        mean = np.mean(semantic_slice, axis=-1, keepdims=True)
        var = np.var(semantic_slice, axis=-1, keepdims=True)
        res[:, semantic_indices] = (semantic_slice - mean) / np.sqrt(var + eps)
    return res


def clip_state(state):
    """Clamps semantic and conflict registers to [-2.0, 2.0], preserving provenance metadata."""
    res = state.copy()
    semantic_indices = _get_semantic_indices()
    if res.ndim == 1:
        for i in semantic_indices:
            res[i] = np.clip(res[i], -2.0, 2.0)
    else:
        for i in semantic_indices:
            res[:, i] = np.clip(res[:, i], -2.0, 2.0)
    return res

class CNVMRuntime:
    def __init__(self, compiled_checkpoint):
        self.embedding = compiled_checkpoint["embedding"]
        self.token_to_id = compiled_checkpoint["token_to_id"]
        self.id_to_token = compiled_checkpoint["id_to_token"]
        self.W_q = compiled_checkpoint["W_q"]
        self.W_k = compiled_checkpoint["W_k"]
        self.W_v = compiled_checkpoint["W_v"]
        self.serg = compiled_checkpoint["serg"]
        self.architecture = compiled_checkpoint.get("architecture", {
            "max_layers": 40,
            "cce_layers": [30, 31, 32, 33, 34],
            "default_cce_max_iter": 10,
            "default_cce_epsilon": 0.1
        })
        
        # Scaling factor for state updates
        self.gamma = 1.0
        
    def get_token_vector(self, token):
        """Looks up a token and returns its 100-dimensional register vector."""
        if token not in self.token_to_id:
            raise KeyError(f"Token '{token}' not found in compiled vocabulary.")
        token_id = self.token_to_id[token]
        return self.embedding[token_id].copy()

    def execute_attention(self, H):
        """
        Executes block-diagonal multi-head self-attention on hidden states H.
        H: np.ndarray of shape (N, 100) or (100,)
        """
        is_flat = H.ndim == 1
        if is_flat:
            H_seq = H[np.newaxis, :]  # shape (1, 100)
        else:
            H_seq = H.copy()

        # Project Q, K, V
        Q = np.dot(H_seq, self.W_q)  # (N, 100)
        K = np.dot(H_seq, self.W_k)  # (N, 100)
        V = np.dot(H_seq, self.W_v)  # (N, 100)

        # Attention weights (using pre-scaled Q/K from compilation)
        logits = np.dot(Q, K.T)  # (N, N)
        
        # Apply ALiBi Asymmetric Bias and Local Sliding Window Mask
        N = H_seq.shape[0]
        if N > 1:
            indices = np.arange(N)
            distance = indices[None, :] - indices[:, None]  # shape (N, N)
            
            # Asymmetric Bias: breaks N > M vs N < M symmetry
            alibi_bias = 0.1 * distance
            logits = logits + alibi_bias
            
            # Local Sliding Window Cutoff: hard constraint from engineering spec
            mask = np.abs(distance) > 5
            logits[mask] = -np.inf
            
        # Softmax over last dimension for sequence attention routing
        max_logits = np.max(logits, axis=-1, keepdims=True)
        # Avoid warnings with -inf by zeroing out the masked exp directly
        exp_logits = np.exp(logits - max_logits)
        if N > 1:
            exp_logits[mask] = 0.0
        attn_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        O = np.dot(attn_weights, V)  # (N, 100)
        
        if is_flat:
            return O[0]
        return O

    def execute_serg_layer(self, H, layer_index):
        """
        Evaluates rules at a given layer index.
        H: np.ndarray of shape (N, 100) or (100,)
        Returns:
            next_state: np.ndarray of the same shape
            active_rules: list of dicts detailing rule triggers and activations
        """
        if layer_index not in self.serg:
            return H.copy(), []

        layer_data = self.serg[layer_index]
        W_in = layer_data["W_in"]      # (100, R)
        W_out = layer_data["W_out"]    # (R, 100)
        b = layer_data["b"]            # (R,)
        rule_ids = layer_data["rule_ids"]
        
        is_flat = H.ndim == 1
        H_seq = H[np.newaxis, :] if is_flat else H.copy()
        
        # Calculate precondition activations: a = ReLU(H * W_in + b)
        # H_seq: (N, 100), W_in: (100, R), b: (R,)
        pre_act = np.dot(H_seq, W_in) + b  # (N, R)
        activations = np.maximum(0.0, pre_act)  # (N, R)
        
        # Calculate updates: Y = gamma * (activations * W_out)
        updates = self.gamma * np.dot(activations, W_out)  # (N, 100)
        next_H = H_seq + updates
        
        # Track active rules for debugging/provenance
        active_rules = []
        for token_idx in range(H_seq.shape[0]):
            token_active = []
            for r_idx in range(W_in.shape[1]):
                act_val = activations[token_idx, r_idx]
                if act_val > 0:
                    token_active.append({
                        "rule_id": rule_ids[r_idx],
                        "activation": float(act_val),
                        "trigger_val": float(H_seq[token_idx, np.argmax(np.abs(W_in[:, r_idx]))])
                    })
            active_rules.append(token_active)
            
        if is_flat:
            return next_H[0], active_rules[0]
        return next_H, active_rules

    def execute_cce_layer(self, H, layer_index, max_iter=10, epsilon=0.1):
        """
        Executes CCE Cognitive Resonance Loop on layer_index.
        Recursively loops routing (DRF) and rule parsing (SERG) until conflict decays.
        """
        conflict_offset = get_register_offset("SYS::CONFLICT")
        h_layer = H.copy()
        
        history = []
        iteration = 0
        converged = False
        
        while iteration < max_iter:
            # 1. Normalize and Route
            h_norm = layer_norm(h_layer)
            routed = h_layer + self.execute_attention(h_norm)
            routed = clip_state(routed)
            
            # 2. Normalize and Evaluate Rules
            routed_norm = layer_norm(routed)
            next_state, active_rules = self.execute_serg_layer(routed_norm, layer_index)
            
            # Residual add from routed (not routed_norm to preserve stream)
            next_state = routed + (next_state - routed_norm)
            next_state = clip_state(next_state)
            
            # Read conflict values
            is_flat = next_state.ndim == 1
            if is_flat:
                conflict_val = float(next_state[conflict_offset])
            else:
                conflict_val = float(np.max(next_state[:, conflict_offset]))
                
            history.append({
                "iteration": iteration,
                "conflict_value": conflict_val,
                "state": next_state.copy(),
                "active_rules": active_rules
            })
            
            h_layer = next_state
            
            if iteration > 0 and conflict_val < epsilon:
                converged = True
                break
                
            iteration += 1
            
        return h_layer, {
            "converged": converged,
            "iterations": iteration + 1,
            "history": history
        }

    def run_forward(self, tokens, max_layer=None, cce_layers=None, max_iter=None, epsilon=None):
        max_layer = max_layer if max_layer is not None else self.architecture.get("max_layers", 40)
        cce_layers = cce_layers if cce_layers is not None else self.architecture.get("cce_layers", [30, 31, 32, 33, 34])
        max_iter = max_iter if max_iter is not None else self.architecture.get("default_cce_max_iter", 10)
        epsilon = epsilon if epsilon is not None else self.architecture.get("default_cce_epsilon", 0.1)
        """
        Runs the full forward execution pass of the CNVM for a list of tokens.
        tokens: list of token strings or list of token ids.
        """
        # Resolve tokens to initial states
        states = []
        for t in tokens:
            if isinstance(t, str):
                states.append(self.get_token_vector(t))
            else:
                states.append(self.embedding[t].copy())
                
        H = np.array(states, dtype=np.float32)  # shape (N, dim)
        
        # Inject positional encoding (0, 1, 2, ... sequence indices)
        pos_offset = get_register_offset("SYNTAX::POSITION_INDEX")
        for idx in range(H.shape[0]):
            H[idx, pos_offset] = float(idx)
            
        trace = []
        for l in range(max_layer):
            if l in cce_layers:
                # Execute CCE iteration loop on this layer
                H, cce_info = self.execute_cce_layer(H, l, max_iter=max_iter, epsilon=epsilon)
                trace.append({
                    "layer": l,
                    "type": "CCE",
                    "cce_info": cce_info
                })
            else:
                # Skip executing standard blocks if they contain no rules, treating them as pass-through
                if l not in self.serg:
                    trace.append({
                        "layer": l,
                        "type": "STANDARD",
                        "active_rules": []
                    })
                    continue
                
                # Regular layer forward pass (executed only if rules are present)
                H_norm = layer_norm(H)
                # Compute attention manually to intercept attn_weights and V
                Q = np.dot(H_norm, self.W_q)
                K = np.dot(H_norm, self.W_k)
                V = np.dot(H_norm, self.W_v)
                logits = np.dot(Q, K.T)
                
                N_seq = H_norm.shape[0]
                if N_seq > 1:
                    indices = np.arange(N_seq)
                    distance = indices[None, :] - indices[:, None]
                    alibi_bias = 0.1 * distance
                    logits = logits + alibi_bias
                    mask = np.abs(distance) > 5
                    logits[mask] = -np.inf
                    
                max_logits = np.max(logits, axis=-1, keepdims=True)
                exp_logits = np.exp(logits - max_logits)
                if N_seq > 1:
                    exp_logits[mask] = 0.0
                attn_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                O = np.dot(attn_weights, V)
                
                routed = H + O
                routed = clip_state(routed)
                
                routed_norm = layer_norm(routed)
                next_H, active_rules = self.execute_serg_layer(routed_norm, l)
                H = routed + (next_H - routed_norm)
                H = clip_state(H)
                
                trace.append({
                    "layer": l,
                    "type": "STANDARD",
                    "active_rules": active_rules
                })
                
        return H, trace
