# Compiled Neural Virtual Machine (CNVM) Compiler & Simulator

![alt text](image.png)

## 🧠 What is CNVM and Why Do We Need It?
Traditional Large Language Models (LLMs) are **stochastic black boxes**. While powerful, they are prone to:
* **Hallucinations:** Predicting words based on statistical probability rather than hard logical constraints.
* **Non-determinism:** Yielding different outputs for the same query depending on temperature and decoding noise.
* **Black-Box Reasoning:** Making it impossible to audit *exactly* why a network reached a specific conclusion.
* **High Compute Overhead:** Processing simple reasoning tasks through billions of parameters unnecessarily.

**CNVM (Compiled Neural Virtual Machine)** solves this by treating the transformer/neural network as a compiled virtual machine. Rather than training weights through trial-and-error (gradient descent), we **directly compile** semantic taxonomies, domain isolation boundaries, and deterministic logic rules into the model's weight matrices.

> [!IMPORTANT]
> **Is this still Artificial Intelligence?**
> Yes. CNVM uses the **exact same architecture** as standard LLMs (including embedding tables, self-attention multi-head matrices, layer normalization, and Feed-Forward Neural network layers). The only difference is the **weight inception method**: instead of relying on stochastic training (Gradient Descent) on massive raw datasets, the weights are algebraically compiled to enforce exact mathematical guarantees. It is a genuine, safety-hardened Neural Network AI.

### What it will be at the end
At maturity, CNVM will serve as a **formally verifiable cognitive processor** for safety-critical systems. It is designed to be embedded in environments where stochastic failure is a liability—such as automated medical diagnostics, aerospace control loops, nuclear power management, and real-time compliance checking.

### Real-World Use Cases
* **Aerospace & Mechanical Repair Diagnostics:** Ensuring a diagnostics assistant strictly checks safety boundaries (e.g. verifying that power is turned off before touching electrical components) and tracks mechanical gear/transmission stress limits with absolute precision.
* **Home Assistant Robotics:** Hardcoding domestic rules directly into the robot's reasoning loops (e.g., "never place a microfiber cloth in the microwave" or "never wash leather seats with an acidic clearcoat cleaner").
* **Scientific & Thermodynamic Modeling:** Modeling chemical, thermal, and biological state boundaries (e.g. yeast fermentation heat ranges, steak internal doneness, or soufflé baking limits) directly inside a high-dimensional vector space without float drift.
* **Factual Knowledge Base Retrieval:** Ensuring absolute recall and zero hallucination when retrieving structured historical records (e.g. national symbols, capital locations, or prime minister directories).
* **Verifiable Compliance & Programming:** Tracking system execution paths where logical transitions must be 100% auditable.

---

## ⚖️ How is CNVM Different from Regular LLMs?

| Feature | Regular LLMs (e.g. GPT-4, LLaMA) | Compiled Neural Virtual Machine (CNVM) |
| :--- | :--- | :--- |
| **Weight Inception** | Stochastic Gradient Descent (SGD) training | Direct algebraic compilation into weight matrices |
| **Determinism** | Probabilistic (can vary per run) | 100% Deterministic (same inputs always yield same vectors) |
| **Hallucination** | Frequent / Unpredictable | Mathematically Impossible (execution bounded by rule graph) |
| **Domain Control** | Soft prompts / fine-tuning (leaky) | Block-diagonal matrix isolation (DRF) + explicit attention bridges |
| **Logic Reasoning** | Emergent / Statistical heuristics | Explicit Sparse Executable Rule Graphs (SERG) & constraint checker |
| **Inference Auditing** | Impossible (requires probing vectors) | Trivial (explicitly writes rule IDs to `META::PROVENANCE`) |

---

## ❓ Critical Q&A (Frequently Asked Questions)

### Q1: If CNVM is deterministic and compiled, why not just write standard code?
Traditional code (like nested `if-else` loops) is rigid, struggles with high-dimensional associative memory, and cannot easily perform fuzzy semantic matching. CNVM embeds symbolic rules directly inside a high-dimensional vector space. This allows it to inherit properties dynamically (e.g. a `sprocket` inheriting properties of a `gearset` automatically) and perform soft-matching on vocabulary while executing rigid logical constraints.

### Q2: Does CNVM require training on GPUs?
No. There is **no training phase** in CNVM. Weight compilation runs instantly in milliseconds on a CPU by solving algebraic definitions. The compiled weights are then injected directly into standard attention and FFN layers.

### Q3: How does the network handle contradictions or conflicts?
CNVM features a **Constraint Convergence Engine (CCE)**. If an input query triggers rules that contradict one another (e.g. a temperature register spiking while a safety audit demands low heat), the system detects a conflict. The CCE runs a recursive loop over the memory states, adjusting weights back down until the contradiction is resolved below a threshold $\epsilon$ and the network stabilizes/halts.

### Q4: Can it generalize to unseen vocabulary?
Yes, via **Ontological Concept Embeddings (ESC)**. Instead of requiring training examples for every possible term, words inherit semantic properties dynamically from parent classes defined in a structured taxonomy. If the system knows `steak` is a Noun and inherits from `Organic Matter`, any new word designated as a type of `steak` automatically inherits all corresponding thermodynamic and cooking properties.

---

A high-fidelity Python implementation and TypeScript simulator for the **Compiled Neural Virtual Machine (CNVM)** architecture. This computing paradigm compiles structured configurations, rules, and token vocabulary maps directly into static neural weight matrices (embedding tables, block-diagonal attention layers, and FFN gates), replacing unconstrained stochastic training with formal constraint convergence logic.

## 🚀 Core Capabilities

1. **Ontological Concept Embeddings (ESC)**: Resolves vocabulary inheritance models dynamically. Sub-concepts (e.g. `sprocket`) inherit baseline domain features from their parent classes (e.g. `gearset`) and clamp them to safe activation ranges.
2. **Domain Isolation via Block-Diagonal Fabric (DRF)**: Isolates semantic domains (e.g. mechanical engineering, astrophysics, security) using block-diagonal projections. Punching holes in this isolation enables Sparse Inter-Domain Bridges (SIDBs) for sequence-based cross-attention routing.
3. **Sparse Executable Rule Graphs (SERG)**: Compiles logical rules into neural gate projections. SERG FFN weights act as precondition detectors (trigger keys) and postcondition effects, writing metadata IDs to rule provenance trackers.
4. **Constraint Convergence Engine (CCE)**: A cognitive loop that executes routing and rule evaluation iteratively. In the presence of logical conflicts (e.g. high stress indicators spiking a monitor), the CCE runs loops recursively to resolve the contradiction below threshold $\epsilon$ and halt.
5. **Interactive Sandbox Web UI**: A Vite + React + TypeScript visualizer providing a grid view of all 100 hidden state coordinates, custom sequence inputs, step-by-step layer traces, and a live manifest compiler interface.

---

## 📂 Project Directory Structure

```text
aiCompiler/
├── manifest/                     # Model configuration schema JSON files
│   ├── sliders.json              # Source of truth coordinate mapping (dimensions & strides)
│   ├── vocabulary.json           # Word embeddings configuration with parent inheritance
│   ├── output_rules.json         # Output token projection targets
│   └── layers/
│       └── layer_7/
│           ├── RULE_*.json       # One FFN rule per JSON file (e.g. stress rules, audits)
│           └── *_Query_*.json    # Attention bridge definitions (e.g. MECHANICAL_ENG_Query -> SYSTEM_ENTROPY_Key)
├── cnvm/                         # Python Core Engine
│   ├── tsr.py                    # Dynamic tensor register slicing & formatting
│   ├── nir.py                    # Manifest parser, schema builder & validator
│   ├── compiler.py               # ESC, DRF, and SERG weight compilers
│   └── runtime.py                # Runtime execution engine (LayerNorm, attention, CCE loop)
├── tests/                        # Mathematical verification suite
│   └── test_verification.py      # Assertions for determinism, locality, convergence, and boundaries
└── dashboard/                    # React Web Simulator
    ├── src/
    │   ├── cnvm/                 # TypeScript port of compiler, runtime & glob manifest loader
    │   ├── App.tsx               # Main dashboard UI component
    │   └── index.css             # Premium glassmorphism dark-theme styling
    ├── package.json              # Web app scripts and dependencies
    └── vite.config.ts            # Vite server configurations allowing parent manifest paths
```

---

## 💻 How-To Guide & CLI Commands

### 1. Prerequisites
Ensure you have the following installed on your system:
* **Python 3.12+** (with `numpy` and `pytest` packages)
* **Node.js 18+** and **npm**

---

### 2. Run Python Mathematical Verification Tests
Formally test determinism, update locality, CCE halting convergence, and bounded activations against the manifest configurations:

```bash
# Execute from the project's root folder:
PYTHONPATH=. pytest tests/
```

Expected output:
```text
============================== 4 passed in 0.15s ===============================
```

---

### 3. Run the Web UI Dashboard Simulator
Follow these commands to install dependencies, run the development server, or build a production bundle.

#### Install Node Dependencies
```bash
# Navigate to the dashboard directory:
cd dashboard

# Install necessary libraries (React, TypeScript, Lucide icons, Vite):
npm install
```

#### Start Local Development Server
Launch the local server with hot module replacement (HMR) enabled:
```bash
# Start the local development server:
npm run dev
```

* This will launch a server, usually accessible at **`http://localhost:5173`**.
* Any edits you make to files under the `manifest/` directory will automatically update inside the running web dashboard!

#### Build Dashboard for Production
To bundle and compile the React code into optimized, minified static files:
```bash
# Build the production files:
npm run build
```
The compiled output will be generated inside `dashboard/dist/`.

---

## ⚙️ Modifying Manifest Settings

The CNVM implementation uses JSON configurations to dictate behavior, completely avoiding hardcoded properties in source files.

* **Add a new word token**: Edit `manifest/vocabulary.json` and optionally inherit from a parent class or declare coordinate activations.
* **Add a SERG FFN Rule**: Create a new `.json` file inside `manifest/layers/layer_N/` starting with `RULE_`. Define the rule ID, target trigger register, outcome register, and input/output weights.
* **Add an Attention Bridge**: Create a new `.json` file inside `manifest/layers/layer_N/` named `SOURCE_DOMAIN_TARGET_DOMAIN.json` (e.g. `COMPUTER_SECURITY_SYSTEM_ENTROPY.json`) and define the query, key, and value weights.
