# Compiled Neural Virtual Machine (CNVM) Architecture Blueprint

We should follow this architecture as a guiding principle, but we are free to adapt as needed, as long as we maintain the core principles. 

To map a 40-layer Compiled Neural Virtual Machine (CNVM), we must assign a single, atomic, mathematically distinct responsibility to every single layer. In standard models, Layer 12 and Layer 32 might do the exact same messy mix of grammar and logic. In our architecture, every layer has a strict, hardcoded job.

---

## ⚙️ Core Mechanisms Explained

Before reading the layer blueprint, it is critical to understand the two primary mathematical mechanisms used to implement these layers:

1. **DRF (Dynamic Routing Fabric / QKV Attention):**
   * **How it works:** This is the routing mechanism. It allows tokens (words) to talk to each other and pass information across the sequence.
   * **Query (Q):** A token projects a query vector searching for a specific property (e.g., a verb looking for its object).
   * **Key (K):** A token projects a key vector broadcasting its own properties (e.g., a noun broadcasting "I am an object").
   * **Value (V):** If a Query and Key match, the Value vector acts as the payload, copying specific slider values from the matching key token to the querying token.

2. **SERG (Sparse Executable Rule Graphs / FFN Gates):**
   * **How it works:** This is the logic-gate mechanism. It acts as an `if-then` rule processor within a single token's hidden state.
   * **Trigger:** The input checks if specific slider coordinates are above a set activation threshold (e.g., `FRICTION > 0.8`).
   * **Gate Out:** If the threshold is crossed, the feed-forward network scales and writes a value to a target slider coordinate (e.g., setting `SYSTEM_ENTROPY` to `-0.5`).

---

## 🗺️ Exhaustive Layer-by-Layer Blueprint

> [!IMPORTANT]
> The detailed, layer-by-layer blueprint (Layers 1-40) has been extracted to a centralized, machine-readable JSON format. 
> 
> You can find the exhaustive list of all 40 layers, including their exact mechanisms, descriptions, teacher extraction prompts, and matrix examples in:
> `manifest/architecture.json`
>
> All system components, including the Python runtime and the TypeScript dashboard, now dynamically load their layer specifications from this single source of truth.