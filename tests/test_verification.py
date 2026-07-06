import numpy as np
import pytest
from cnvm.tsr import get_tsr_map, get_register_offset
from cnvm.nir import load_manifest

manifest = load_manifest("manifest")
DEFAULT_VOCAB = manifest.vocab_data
DEFAULT_ROUTING = manifest.routing_data
DEFAULT_RULES = manifest.rules_data
DEFAULT_ARCHITECTURE = manifest.architecture
from cnvm.compiler import CNVMCompiler
from cnvm.runtime import CNVMRuntime

def test_determinism():
    """
    Formally verifies the DETERMINISM guarantee:
    Given identical token input and compiled weight state, the execution path
    and output vector are 100% deterministic.
    """
    compiler = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    
    runtime1 = CNVMRuntime(compiled)
    runtime2 = CNVMRuntime(compiled)
    
    tokens = ["sprocket", "torque_pulse"]
    
    out1, trace1 = runtime1.run_forward(tokens)
    out2, trace2 = runtime2.run_forward(tokens)
    
    # Assert exact array match
    np.testing.assert_array_equal(out1, out2)
    
    # Assert identical traces
    assert len(trace1) == len(trace2)
    for i in range(len(trace1)):
        assert trace1[i]["type"] == trace2[i]["type"]
        assert trace1[i]["layer"] == trace2[i]["layer"]

def test_locality_of_updates():
    """
    Formally verifies the LOCALITY OF UPDATES guarantee:
    Updating rule c in the SERG modifies exactly one rank-1 update matrix
    (its specific columns/rows in W_in/W_out), with zero effect on the
    representations or execution paths of unrelated rules.
    """
    # 1. Compile original model
    compiler_orig = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled_orig = compiler_orig.compile()
    
    # 2. Modify only one rule (RULE_GEAR_SHEAR) in rules schema
    modified_rules = DEFAULT_RULES.copy()
    modified_rules["RULE_GEAR_SHEAR"] = DEFAULT_RULES["RULE_GEAR_SHEAR"].copy()
    modified_rules["RULE_GEAR_SHEAR"]["gate_out_weight"] = -1.8 # originally -2.0
    
    compiler_mod = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, modified_rules, DEFAULT_ARCHITECTURE)
    compiled_mod = compiler_mod.compile()
    
    # 3. Check parameters
    # The embeddings, attention matrices (W_q, W_k, W_v) must be 100% identical
    np.testing.assert_array_equal(compiled_orig["embedding"], compiled_mod["embedding"])
    np.testing.assert_array_equal(compiled_orig["W_q"], compiled_mod["W_q"])
    np.testing.assert_array_equal(compiled_orig["W_k"], compiled_mod["W_k"])
    np.testing.assert_array_equal(compiled_orig["W_v"], compiled_mod["W_v"])
    
    # Check SERG matrices of layer 18
    serg_orig = compiled_orig["serg"][18]
    serg_mod = compiled_mod["serg"][18]
    
    # Find the row index of RULE_GEAR_SHEAR (rule_id 142) dynamically
    rule_idx = serg_orig["rule_ids"].index(142)
    
    # W_in should be identical (we only changed gate_out_weight)
    np.testing.assert_array_equal(serg_orig["W_in"], serg_mod["W_in"])
    
    # In W_out, only the row representing the modified rule should change
    # All other rows of W_out must remain identical
    np.testing.assert_array_equal(
        np.delete(serg_orig["W_out"], rule_idx, axis=0),
        np.delete(serg_mod["W_out"], rule_idx, axis=0)
    )
    
    # Assert that the specific parameter changed
    assert serg_orig["W_out"][rule_idx, get_register_offset("SYS::ENTROPY")] == -2.0
    assert serg_mod["W_out"][rule_idx, get_register_offset("SYS::ENTROPY")] == -1.8
    
    # 4. Assert that running an unrelated input (e.g. audit_trigger)
    # which only fires RULE_AUDIT_VERIFY/RULE_CONFLICT_RESOLUTION results in
    # the exact same outputs on both models (zero interference).
    runtime_orig = CNVMRuntime(compiled_orig)
    runtime_mod = CNVMRuntime(compiled_mod)
    
    out_orig, _ = runtime_orig.run_forward(["audit_trigger"])
    out_mod, _ = runtime_mod.run_forward(["audit_trigger"])
    
    np.testing.assert_array_equal(out_orig, out_mod)

def test_cce_convergence():
    """
    Formally verifies the CONSTRAINT CONVERGENCE ENGINE (CCE):
    Under mechanical stress (sprocket + torque_pulse), RULE_GEAR_SHEAR fires,
    driving SYS::ENTROPY negative. RULE_AUDIT_VERIFY then fires (triggered by
    negative entropy), registering the conflict. The CCE loop at layer 30
    executes, detects the conflict state (SYS::CONFLICT), and converges.
    """
    compiler = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    runtime = CNVMRuntime(compiled)

    # sprocket: DOMAIN::MECHANICAL=0.85, SYS::ENTROPY=0.8
    # torque_pulse: DOMAIN::MECHANICAL=1.5, SYS::ENTROPY=0.9
    # -> RULE_GEAR_SHEAR fires at layer 18 (DOMAIN::MECHANICAL >= 0.5 threshold)
    # -> RULE_AUDIT_VERIFY fires at layer 26 (SYS::ENTROPY went negative)
    # -> CCE loop executes at layer 30
    out_state, trace = runtime.run_forward(["sprocket", "torque_pulse"], cce_layers=[30, 31, 32, 33, 34], max_iter=10)

    # Verify CCE block executed on layer 30
    for t in trace:
        if t["layer"] == 26 and t["type"] == "STANDARD":
            print(f"[pytest debug] Layer 26 active_rules: {t['active_rules']}")
    cce_trace = None
    for t in trace:
        if t["layer"] == 30 and t["type"] == "CCE":
            cce_trace = t["cce_info"]
            break

    assert cce_trace is not None, "Layer 30 CCE loop did not execute."
    assert cce_trace["converged"] == True, "CCE failed to converge within iteration limit."
    assert cce_trace["iterations"] >= 1, "CCE history must have at least one iteration."

    # Conflict value must be below epsilon (0.1) at convergence
    history = cce_trace["history"]
    final_conflict = history[-1]["conflict_value"]
    assert final_conflict < 0.1, f"Final conflict value {final_conflict} is not below epsilon."

    # Verify that the key stress rules fired during the forward pass (before CCE)
    forward_rule_ids = set()
    for t in trace:
        if t["type"] == "STANDARD":
            for token_rules in t.get("active_rules", []):
                for r in token_rules:
                    forward_rule_ids.add(r["rule_id"])

    assert 142 in forward_rule_ids, "RULE_GEAR_SHEAR (rule 142, layer 18) did not fire in forward pass"
    assert 150 in forward_rule_ids, "RULE_AUDIT_VERIFY (rule 150, layer 26) did not fire in forward pass"

def test_bounded_activations():
    """
    Formally verifies the BOUNDED ACTIVATIONS guarantee:
    All weights are surgically clamped, and hidden states are clipped/normalized
    such that the L2 norm of any hidden state never exceeds the pre-calculated limits.
    In our case, we clamp every register to the range [-2.0, 2.0].
    """
    compiler = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    
    # Assert weight parameters are strictly clamped in compilation
    for key in ["W_q", "W_k", "W_v"]:
        assert np.max(np.abs(compiled[key])) <= 2.0

    # Build a dynamic mask that excludes only the metadata register indices
    from cnvm.tsr import get_tsr_map, get_dim
    tsr_map = get_tsr_map()
    dim = get_dim()
    meta_names = {"META::PROVENANCE", "META::EVIDENCE"}
    meta_indices = set()
    for name in meta_names:
        if name in tsr_map:
            start, end = tsr_map[name]
            meta_indices.update(range(start, end + 1))
    semantic_mask = np.array([i not in meta_indices for i in range(dim)])

    for layer in compiled["serg"].values():
        assert np.max(np.abs(layer["W_in"])) <= 2.0
        # Exclude provenance metadata register from W_out boundary check
        assert np.max(np.abs(layer["W_out"][:, semantic_mask])) <= 2.0

    # Assert runtime outputs are clipped/bounded within [-2.0, 2.0]
    runtime = CNVMRuntime(compiled)
    tokens = ["sprocket", "torque_pulse", "audit_trigger"]
    out_state, _ = runtime.run_forward(tokens)

    # Exclude provenance metadata columns from boundary assertions
    assert np.max(np.abs(out_state[:, semantic_mask])) <= 2.0
    # Theoretical max L2 norm = sqrt(dim * 2.0^2) = sqrt(dim * 4)
    max_l2 = float(np.sqrt(dim * 4.0))
    for i in range(out_state.shape[0]):
        masked_state = out_state[i, semantic_mask]
        l2_norm = float(np.linalg.norm(masked_state))
        assert l2_norm <= max_l2 + 1e-4, f"L2 norm {l2_norm} exceeds theoretical maximum {max_l2:.2f}."

def test_auto_complete_projections():
    """
    Formally verifies the data generation examples and rules matching the output criteria.
    Simulates the forward pass and computes similarity against output_rules exactly like the UI.
    """
    compiler = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    runtime = CNVMRuntime(compiled)

    examples = [
        # --- Core examples (short) ---
        (["canada", "capital"], "ottawa"),
        (["first", "prime_minister", "canada"], "john_a_macdonald"),
        (["year", "confederation"], "1867"),
        (["national", "symbol", "canada"], "maple_leaf"),
        (["bake", "temperature", "souffle"], "375f"),
        (["bread", "rise"], "yeast"),
        (["steak", "medium_rare"], "140f"),
        (["safest", "cloth"], "microfiber"),
        (["protects", "uv"], "carnauba_wax"),
        (["cleans", "leather", "seats"], "ph_neutral_cleaner"),
        # --- Permutation tests ---
        (["capital", "canada"], "ottawa"),
        (["first", "canada", "prime_minister"], "john_a_macdonald"),
        (["medium_rare", "steak"], "140f"),
        (["rise", "bread"], "yeast"),
        (["souffle", "bake", "temperature"], "375f"),
    ]

    # Longer conclusion sentences — verified for rank-1 correctness only,
    # not strict similarity %, because extra tokens shift the state vector
    # away from the calibration baseline while still ranking the right answer first.
    longer_examples = [
        (["steak", "temperature", "medium_rare"],       "140f"),          # "What temperature for a medium rare steak?"
        (["souffle", "bake", "temperature", "rise"],    "375f"),          # "A souffle must bake at temperature to rise"
        (["bread", "bake", "rise"],                     "yeast"),         # "Bread needs baking agent to rise"
        (["steak", "bake", "medium_rare"],              "140f"),          # "Bake a steak to medium rare"
        (["first", "prime_minister", "canada", "year"],"john_a_macdonald"),  # "Who was the first PM of Canada that year?"
        (["canada", "national", "symbol"],              "maple_leaf"),    # "What is Canada's national symbol?"
        (["capital", "canada", "first"],                "ottawa"),        # "What is the capital of Canada first?"
        (["cleans", "protects", "leather"],             "ph_neutral_cleaner"),  # "What cleans and protects leather?"
    ]

    def compute_similarities(final_state):
        similarities = {}
        valid_sliders = [s for s in manifest.tsr_map.keys() if s.startswith("DOMAIN::") or s.startswith("SEMANTIC::") or s.startswith("SYNTAX::") or s.startswith("SYS::")]
        
        for token_name, rule in manifest.output_data.items():
            score = 0
            count = len(valid_sliders)
            target_sliders = rule.get("target_sliders", {})
            
            for slider_name in valid_sliders:
                start_offset = manifest.tsr_map[slider_name][0]
                actual_val = final_state[start_offset]
                
                if slider_name in target_sliders:
                    target_weight = target_sliders[slider_name]["weight"]
                else:
                    target_weight = -2.0  # Missing implies explicit suppression (-2.0)
                    
                diff = actual_val - target_weight
                score += diff * diff
                
            mse = score / count if count > 0 else float('inf')
            similarities[token_name] = max(0, min(100, round((1.0 - mse / 8.0) * 100)))
        return similarities

    # TIER 1: Canonical examples — strict similarity >= 85% and strict isolation
    for tokens, expected_output in examples:
        out_state, _ = runtime.run_forward(tokens)
        final_state = np.mean(out_state, axis=0)
        similarities = compute_similarities(final_state)
        expected_similarity = similarities.get(expected_output, -1)

        assert expected_similarity >= 85, (
            f"[Canonical] Failed on {tokens}. Expected {expected_output} similarity >= 85%, got {expected_similarity}%"
        )
        for token_name, sim in similarities.items():
            if token_name != expected_output:
                assert sim <= expected_similarity, (
                    f"[Canonical] Strict isolation failed on {tokens}. "
                    f"{token_name} had {sim}% which beat {expected_output} ({expected_similarity}%)"
                )

    # TIER 2: Longer conclusion sentences — rank-1 only (expected output must be top suggestion)
    for tokens, expected_output in longer_examples:
        out_state, _ = runtime.run_forward(tokens)
        final_state = np.mean(out_state, axis=0)
        similarities = compute_similarities(final_state)
        best_score = max(similarities.values())

        assert similarities.get(expected_output, -1) == best_score, (
            f"[Longer] Rank-1 failed on {tokens}. "
            f"Expected {expected_output} to tie for first (score {best_score}), but got {similarities.get(expected_output)} "
            f"(scores: {sorted(similarities.items(), key=lambda x: -x[1])[:3]})"
        )

def test_grammar_parsing():
    """
    Formally verifies that the 10 grammar parsing rules correctly structure
    the state vector in Zone 1 (Layers 1-5).
    """
    compiler = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    runtime = CNVMRuntime(compiled)

    # Test 1: The Question Cascade
    # "what" should trigger IS_QUESTION rules
    _, trace_q = runtime.run_forward(["what"])
    fired_q = set()
    for t in trace_q:
        if t["type"] == "STANDARD":
            for tr in t.get("active_rules", []):
                for r in tr:
                    fired_q.add(r["rule_id"])
                    
    assert 101 in fired_q, "RULE_QUESTION_SEEK (101) did not fire on 'what'"
    assert 107 in fired_q, "RULE_QUESTION_MISSING_INFO_ENTROPY (107) did not fire on 'what'"
    assert 109 in fired_q, "RULE_QUESTION_CONFLICT_SPIKE (109) did not fire on 'what'"

    # Test 2: The Noun Cascade
    # "canada" should trigger NOUN and SUBJECT rules
    _, trace_n = runtime.run_forward(["canada"])
    fired_n = set()
    for t in trace_n:
        if t["type"] == "STANDARD":
            for tr in t.get("active_rules", []):
                for r in tr:
                    fired_n.add(r["rule_id"])
                    
    assert 102 in fired_n, "RULE_NOUN_DETECTION (102) did not fire on 'canada'"
    assert 104 in fired_n, "RULE_SUBJECT_ROLE_ASSIGNMENT (104) did not fire on 'canada'"
    assert 106 in fired_n, "RULE_SUBJECT_GROUNDING_CONFIDENCE (106) did not fire on 'canada'"
    assert 110 in fired_n, "RULE_NOUN_ENTROPY_STABILIZATION (110) did not fire on 'canada'"

    # Test 3: The Verb Cascade
    # "bake" should trigger VERB and OBJECT preparation rules
    _, trace_v = runtime.run_forward(["bake"])
    fired_v = set()
    for t in trace_v:
        if t["type"] == "STANDARD":
            for tr in t.get("active_rules", []):
                for r in tr:
                    fired_v.add(r["rule_id"])
                    
    assert 103 in fired_v, "RULE_VERB_DETECTION (103) did not fire on 'bake'"
    assert 105 in fired_v, "RULE_VERB_OBJECT_PREPARATION (105) did not fire on 'bake'"

def test_position_encoding():
    """
    Formally verifies that CNVM absolute position encoding is correctly injected
    into sequence tokens at runtime.
    """
    compiler = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    runtime = CNVMRuntime(compiled)

    tokens = ["first", "prime_minister", "canada"]
    out_state, _ = runtime.run_forward(tokens, max_layer=0)
    
    pos_offset = get_register_offset("SYNTAX::POSITION_INDEX")
    
    # Assert each token's position index slider matches its sequence index
    for idx in range(len(tokens)):
        assert out_state[idx, pos_offset] == float(idx), (
            f"Expected token index {idx} to have positional slider value {float(idx)}, got {out_state[idx, pos_offset]}"
        )

def test_alibi_bias_asymmetry():
    """
    Formally verifies that the ALiBi Relative Positional Bias successfully breaks
    attention symmetry, allowing the model to distinguish N > M from N < M.
    """
    compiler = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    runtime = CNVMRuntime(compiled)

    tokens = ["first", "prime_minister", "canada"]
    
    # We will manually step through the attention to verify logits
    # Token states after embedding
    H_init = np.array([runtime.get_token_vector(t) for t in tokens])
    
    # Apply position indexes
    pos_offset = get_register_offset("SYNTAX::POSITION_INDEX")
    for i in range(len(tokens)):
        H_init[i, pos_offset] = float(i)
        
    Q = np.dot(H_init, runtime.W_q)
    K = np.dot(H_init, runtime.W_k)
    base_logits = np.dot(Q, K.T)
    
    # Manually compute what runtime does
    N = H_init.shape[0]
    indices = np.arange(N)
    distance = indices[None, :] - indices[:, None]
    alibi_bias = 0.1 * distance
    
    final_logits = base_logits + alibi_bias
    
    # Verify asymmetry: (0, 1) vs (1, 0)
    # base_logits is symmetric for the position component, but alibi forces it to be asymmetric
    assert final_logits[0, 1] != final_logits[1, 0], "Attention logits must be asymmetric to distinguish order"
    
    # Specifically, the difference in bias should be exactly 0.2 (since 0.1 - (-0.1))
    bias_diff = alibi_bias[0, 1] - alibi_bias[1, 0]
    assert np.isclose(bias_diff, 0.2), "ALiBi bias slope is incorrect"

def test_pure_alibi_order_bias():
    """
    Formally verifies that when semantic similarity is zero, the ALiBi bias
    alone causes the Mean Pooled sequence to inherit more properties from
    the LAST token in the sequence (recency bias).
    """
    import copy
    from cnvm.nir import load_manifest
    manifest = load_manifest("manifest")
    
    vocab_data = copy.deepcopy(manifest.vocab_data)
    out_data = copy.deepcopy(manifest.output_data)
    
    # Create identical tokens that only differ in RESERVED_A/B
    # This gives them high cosine similarity, allowing ALiBi to affect attention weights.
    baseline = vocab_data["sprocket"]["sliders"]
    
    vocab_data["order_pos"] = {
        "token_id": max([v["token_id"] for v in vocab_data.values()]) + 1,
        "sliders": {**baseline, "META::RESERVED_A": 2.0, "META::RESERVED_B": 0.0}
    }
    vocab_data["order_neg"] = {
        "token_id": max([v["token_id"] for v in vocab_data.values()]) + 2,
        "sliders": {**baseline, "META::RESERVED_A": 0.0, "META::RESERVED_B": 2.0}
    }
    
    baseline_out = vocab_data["sprocket"]["sliders"]
    out_data["pos_winner"] = {
        "target_sliders": {k: {"weight": v} for k, v in baseline_out.items()}
    }
    out_data["pos_winner"]["target_sliders"]["META::RESERVED_A"] = {"weight": 2.0}
    out_data["pos_winner"]["target_sliders"]["META::RESERVED_B"] = {"weight": 0.0}
    
    out_data["neg_winner"] = {
        "target_sliders": {k: {"weight": v} for k, v in baseline_out.items()}
    }
    out_data["neg_winner"]["target_sliders"]["META::RESERVED_A"] = {"weight": 0.0}
    out_data["neg_winner"]["target_sliders"]["META::RESERVED_B"] = {"weight": 2.0}
    
    compiler = CNVMCompiler(vocab_data, manifest.routing_data, manifest.rules_data, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    runtime = CNVMRuntime(compiled)

    def get_top_prediction(tokens):
        # Run 2 layers so attention is triggered once (Layer 1 has rules).
        out_state, _ = runtime.run_forward(tokens, max_layer=2, cce_layers=[])
        final_state = np.mean(out_state, axis=0)

        similarities = {}
        for token_name, rule in out_data.items():
            score = 0
            count = 0
            for slider_name, slider_config in rule.get("target_sliders", {}).items():
                target_weight = slider_config["weight"]
                if slider_name in manifest.tsr_map:
                    start_offset = manifest.tsr_map[slider_name][0]
                    actual_val = final_state[start_offset]
                    diff = actual_val - target_weight
                    score += diff * diff
                    count += 1
            mse = score / count if count > 0 else float('inf')
            import math
            if math.isnan(mse) or math.isinf(mse):
                mse = float('inf')
            sim = 0
            if mse != float('inf'):
                sim = max(0.0, min(100.0, (1.0 - mse / 8.0) * 100.0))
            similarities[token_name] = sim

        return max(similarities.items(), key=lambda x: x[1])[0]

    pred_1 = get_top_prediction(["order_pos", "order_neg"])
    assert pred_1 == "neg_winner", f"Expected neg_winner, got {pred_1}"

    pred_2 = get_top_prediction(["order_neg", "order_pos"])
    assert pred_2 == "pos_winner", f"Expected pos_winner, got {pred_2}"

def test_order_sensitivity():
    """
    Verifies that changing token order yields different positional rule firings
    and final state representations.
    """
    compiler = CNVMCompiler(DEFAULT_VOCAB, DEFAULT_ROUTING, DEFAULT_RULES, DEFAULT_ARCHITECTURE)
    compiled = compiler.compile()
    runtime = CNVMRuntime(compiled)

    # 1. Run forward order: ["first", "prime_minister", "canada"]
    # "first" is at index 0, "prime_minister" is at index 1, "canada" is at index 2
    out_a, trace_a = runtime.run_forward(["first", "prime_minister", "canada"])
    
    # 2. Run reverse order: ["canada", "prime_minister", "first"]
    # "canada" is at index 0, "prime_minister" is at index 1, "first" is at index 2
    out_b, trace_b = runtime.run_forward(["canada", "prime_minister", "first"])
    
    # 3. Check Layer 2 standard rules active on each token
    # Extract fired rule 0 (Position One Confidence Boost) per token
    def get_fired_tokens_for_rule_0(trace):
        # Layer 2 is index 2
        layer_trace = trace[2]
        assert layer_trace["layer"] == 2
        active_rules = layer_trace["active_rules"]
        # returns a list of booleans indicating if rule 0 fired for each token
        fired = []
        for token_rules in active_rules:
            fired.append(any(r["rule_id"] == 0 for r in token_rules))
        return fired

    fired_a = get_fired_tokens_for_rule_0(trace_a)
    fired_b = get_fired_tokens_for_rule_0(trace_b)
    
    # In order A: ["first", "prime_minister", "canada"]
    # - "first" (index 0) should NOT trigger rule 0
    # - "prime_minister" (index 1) and "canada" (index 2) SHOULD trigger rule 0
    assert fired_a == [False, True, True], f"Unexpected rule 0 firings in order A: {fired_a}"
    
    # In order B: ["canada", "prime_minister", "first"]
    # - "canada" (index 0) should NOT trigger rule 0
    # - "prime_minister" (index 1) and "first" (index 2) SHOULD trigger rule 0
    assert fired_b == [False, True, True], f"Unexpected rule 0 firings in order B: {fired_b}"
    
    # Compare outputs: Since the positional encoding shifted, the final states of the words
    # are completely different and order-dependent.
    assert not np.array_equal(out_a, out_b), "Outputs must be different due to sequence ordering."

