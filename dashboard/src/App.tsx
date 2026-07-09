import { useState, useMemo } from 'react';
import { 
  RotateCcw, 
  AlertTriangle, 
  CheckCircle, 
  Cpu, 
  Settings, 
  Table, 
  Layers, 
  ArrowRight, 
  Info, 
  Save, 
  Sparkles,
  Search,
  BookOpen
} from 'lucide-react';
import { loadEagerManifest, getEagerManifestModules, parseManifest } from './cnvm/loader';
import type { ManifestData } from './cnvm/loader';
import { CNVMCompiler } from './cnvm/compiler';
import type { CompiledCheckpoint } from './cnvm/compiler';
import { CNVMRuntime } from './cnvm/runtime';
import { ExecutionGraph } from './components/ExecutionGraph';

export default function App() {
  // Tabs: 'simulator' | 'editor' | 'weights' | 'knowledge' | 'graph'
  const [activeTab, setActiveTab] = useState<'simulator' | 'editor' | 'weights' | 'knowledge' | 'graph'>('simulator');

  // Load and compile initial manifest
  const [manifest, setManifest] = useState<ManifestData>(() => loadEagerManifest());
  const [compileError, setCompileError] = useState<string | null>(null);

  // File texts for the JSON Schema Editor
  const [fileTexts, setFileTexts] = useState<Record<string, string>>(() => {
    const modules = getEagerManifestModules();
    const texts: Record<string, string> = {};
    for (const [path, module] of Object.entries(modules)) {
      const data = module && (module.default !== undefined ? module.default : module);
      if (data) {
        const cleanKey = path.replace("../../../manifest/", "");
        texts[cleanKey] = JSON.stringify(data, null, 2);
      }
    }
    return texts;
  });

  const [selectedFile, setSelectedFile] = useState<string>(() => {
    const keys = Object.keys(fileTexts);
    return keys.includes("sliders.json") ? "sliders.json" : keys[0] || "";
  });

  // Current compiler checkpoint
  const [checkpoint, setCheckpoint] = useState<CompiledCheckpoint>(() => {
    const initialManifest = loadEagerManifest();
    const compiler = new CNVMCompiler(
      initialManifest.vocabData,
      initialManifest.routingData,
      initialManifest.rulesData,
      initialManifest.tsrMap,
      initialManifest.dim,
      initialManifest.architecture
    );
    return compiler.compile();
  });

  // Simulator State
  const [selectedTokens, setSelectedTokens] = useState<string[]>(["canada", "capital"]);
  const maxLayers = manifest.architecture?.max_layers ?? 40;
  const cceLayers = manifest.architecture?.cce_layers ?? [30, 31, 32, 33, 34];
  const [maxCceIter, setMaxCceIter] = useState<number>(manifest.architecture?.default_cce_max_iter ?? 10);
  const [epsilon, setEpsilon] = useState<number>(manifest.architecture?.default_cce_epsilon ?? 0.1);

  // Simulation execution outcome
  const simulationResult = useMemo(() => {
    if (selectedTokens.length === 0) return null;
    try {
      const runtime = new CNVMRuntime(checkpoint, manifest.tsrMap, manifest.dim);
      return runtime.runForward(selectedTokens, maxLayers, cceLayers, maxCceIter, epsilon);
    } catch (err: any) {
      console.error(err);
      return null;
    }
  }, [checkpoint, manifest, selectedTokens, maxLayers, cceLayers, maxCceIter, epsilon]);

  // Trace selection for visualizer
  const [selectedStepIdx, setSelectedStepIdx] = useState<number>(1); // default to first step or CCE step (layer 7 is index 1 usually)
  const [selectedCceIter, setSelectedCceIter] = useState<number>(2); // default to step inside CCE where rules evaluate
  const [visualizedTokenIdx, setVisualizedTokenIdx] = useState<number>(0);

  // Re-sync trace selection if simulation tokens change
  useMemo(() => {
    if (simulationResult && simulationResult.trace.length > 0) {
      setVisualizedTokenIdx(Math.max(0, selectedTokens.length - 1));
      // Find the CCE trace if exists to make it active by default
      const cceIdx = simulationResult.trace.findIndex(t => t.type === 'CCE');
      if (cceIdx !== -1) {
        setSelectedStepIdx(cceIdx);
        // Find CCE convergence history length
        const cceInfo = simulationResult.trace[cceIdx].cce_info;
        if (cceInfo && cceInfo.history.length > 0) {
          setSelectedCceIter(Math.min(2, cceInfo.history.length - 1));
        }
      } else {
        setSelectedStepIdx(0);
      }
    }
  }, [selectedTokens, checkpoint]);

  // Handle re-compiling schema edits
  const handleCompile = () => {
    try {
      const parsedModules: Record<string, any> = {};
      for (const [key, text] of Object.entries(fileTexts)) {
        const fullPath = "../../../manifest/" + key;
        try {
          parsedModules[fullPath] = JSON.parse(text);
        } catch (jsonErr: any) {
          throw new Error(`JSON Syntax Error in '${key}': ${jsonErr.message}`);
        }
      }

      const newManifest = parseManifest(parsedModules);
      const compiler = new CNVMCompiler(
        newManifest.vocabData,
        newManifest.routingData,
        newManifest.rulesData,
        newManifest.tsrMap,
        newManifest.dim,
        newManifest.architecture
      );
      const newCheckpoint = compiler.compile();

      setManifest(newManifest);
      setCheckpoint(newCheckpoint);
      setCompileError(null);
      alert("CNVM Compiled successfully!");
    } catch (err: any) {
      setCompileError(err.message);
    }
  };

  // Get active state vector based on selected trace step and CCE iteration
  const activeStateVector = useMemo(() => {
    if (!simulationResult) return null;
    const step = simulationResult.trace[selectedStepIdx];
    if (!step) return null;

    if (step.type === 'CCE' && step.cce_info) {
      const histStep = step.cce_info.history[selectedCceIter];
      return histStep ? histStep.state : null;
    } else {
      // Standard layer trace state (final layer outcome)
      // Since trace doesn't explicitly store intermediate state, we can run forward up to this layer
      // to obtain the state vector.
      try {
        const runtime = new CNVMRuntime(checkpoint, manifest.tsrMap, manifest.dim);
        const tokens = selectedTokens;
        const subResult = runtime.runForward(tokens, selectedStepIdx + 1, cceLayers, maxCceIter, epsilon);
        return subResult.finalState;
      } catch {
        return null;
      }
    }
  }, [simulationResult, selectedStepIdx, selectedCceIter, checkpoint, manifest, selectedTokens, cceLayers, maxCceIter, epsilon]);

  // Hovered state coordinate details
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const hoveredDetail = useMemo(() => {
    if (hoveredIdx === null || !manifest) return null;
    const { tsrMap } = manifest;

    // Find which block this index falls into
    for (const [name, [start, end]] of Object.entries(tsrMap)) {
      if (hoveredIdx >= start && hoveredIdx <= end) {
        const offset = hoveredIdx - start;
        let regDesc = "";
        
        const sliderInfo = manifest.slidersConfig[name];
        if (sliderInfo) {
          regDesc = offset === 0 
            ? sliderInfo.description || `${sliderInfo.name} Primary Indicator`
            : `${sliderInfo.name} feature space [slot ${offset}]`;
        } else {
          regDesc = `Coordinate block: ${name} [slot ${offset}]`;
        }

        return {
          name,
          start,
          end,
          offset,
          description: regDesc
        };
      }
    }
    return null;
  }, [hoveredIdx, manifest]);

  // Auto-complete suggestion engine
  const autocompleteSuggestions = useMemo(() => {
    if (!simulationResult || !simulationResult.finalState || simulationResult.finalState.length === 0) {
      return [];
    }
    const finalStateVector = simulationResult.finalState;
    const finalTokenIdx = Math.max(0, selectedTokens.length - 1);
    const tokenState = finalStateVector[finalTokenIdx];

    const results = Object.entries(manifest.outputData).map(([tokenName, rule]) => {
      let score = 0;
      let count = 0;
      const targetSliders = rule.target_sliders || {};

      for (const [sliderName, sliderConfig] of Object.entries(targetSliders)) {
        const targetWeight = (sliderConfig as any).weight;
        const startOffset = manifest.tsrMap[sliderName]?.[0];
        if (startOffset !== undefined) {
          const actualVal = tokenState[startOffset];
          const diff = actualVal - targetWeight;
          score += diff * diff;
          count++;
        }
      }

      const mse = count > 0 ? score / count : Infinity;
      const similarity = Math.max(0, Math.min(100, Math.round((1.0 - mse / 8.0) * 100)));

      return {
        token: tokenName,
        id: rule.token_id,
        description: rule.intent_description,
        similarity,
        mse
      };
    });

    return results.sort((a, b) => b.similarity - a.similarity);
  }, [simulationResult, manifest, selectedTokens]);

  const getCellTooltip = (domain: string, offset: number, val: number): string => {
    const desc = manifest.slidersConfig[domain]?.description || "";
    const name = manifest.slidersConfig[domain]?.name || domain;
    return `${name} (Offset +${offset})\nValue: ${val.toFixed(6)}\n\n${desc}`;
  };

  const ruleMap = useMemo(() => {
    if (!manifest) return {};
    const map: Record<number, any> = {};
    for (const [name, rule] of Object.entries(manifest.rulesData)) {
      if (rule.rule_id !== undefined) {
        map[rule.rule_id] = { name, ...rule };
      }
    }
    return map;
  }, [manifest]);

  return (
    <div id="root">
      {/* Header bar */}
      <header style={{
        padding: '20px 24px',
        borderBottom: '1px solid var(--border-light)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(11, 15, 25, 0.6)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, var(--accent-purple), var(--accent-blue))',
            padding: '8px',
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Cpu size={24} color="white" />
          </div>
          <div>
            <h1>CNVM Neural Virtual Machine</h1>
            <p className="subtitle">Compiled Neural Virtual Machine Architecture Simulator</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div className="glass-panel" style={{ padding: '6px 12px', display: 'flex', gap: '16px', fontSize: '0.85rem' }}>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>DIM:</span>{' '}
              <strong style={{ color: 'var(--accent-cyan)' }}>{manifest.dim}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>VOCAB:</span>{' '}
              <strong style={{ color: 'var(--accent-purple)' }}>{Object.keys(manifest.vocabData).length}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>RULES:</span>{' '}
              <strong style={{ color: 'var(--accent-amber)' }}>{Object.keys(manifest.rulesData).length}</strong>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem' }}>
            <span className="badge badge-emerald" style={{ gap: '4px' }}>
              <CheckCircle size={12} /> Compiled
            </span>
          </div>
        </div>
      </header>

      {/* Tabs Navigator */}
      <div className="app-container" style={{ paddingBottom: 0 }}>
        <div className="tabs-nav">
          <button 
            className={`tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
            onClick={() => setActiveTab('simulator')}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Sparkles size={16} /> Simulator & Visualizer
          </button>
          <button 
            className={`tab-btn ${activeTab === 'editor' ? 'active' : ''}`}
            onClick={() => setActiveTab('editor')}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Settings size={16} /> JSON Manifest Editor
          </button>
          <button 
            className={`tab-btn ${activeTab === 'weights' ? 'active' : ''}`}
            onClick={() => setActiveTab('weights')}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Table size={16} /> Weights Inspector
          </button>
          <button 
            className={`tab-btn ${activeTab === 'knowledge' ? 'active' : ''}`}
            onClick={() => setActiveTab('knowledge')}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <BookOpen size={16} /> Knowledge Base
          </button>
          <button 
            className={`tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
            onClick={() => setActiveTab('graph')}
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Layers size={16} /> Execution Graph Flow
          </button>
        </div>
      </div>

      {/* Main Body content */}
      <main className="app-container" style={{ flex: 1, paddingTop: 0 }}>
        
        {/* TAB 1: SIMULATOR */}
        {activeTab === 'simulator' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '20px' }}>
            
            {/* Left Column: Sequence and Trace */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              
              {/* Sequence Bootstrap Card */}
              <div className="glass-panel" style={{ padding: '20px' }}>
                <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sparkles size={18} color="var(--accent-purple)" /> Input Token Sequence Bootstrapper
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                  Select tokens from the compiled vocabulary to build the sequence. Click a token tag to toggle its inclusion in the active execution context.
                </p>
                
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '20px' }}>
                  {Object.keys(manifest.vocabData).sort().map(token => {
                    const isSelected = selectedTokens.includes(token);
                    return (
                      <button
                        key={token}
                        onClick={() => {
                          if (isSelected) {
                            setSelectedTokens(selectedTokens.filter(t => t !== token));
                          } else {
                            setSelectedTokens([...selectedTokens, token]);
                          }
                        }}
                        style={{
                          background: isSelected 
                            ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(59, 130, 246, 0.3))' 
                            : 'rgba(255, 255, 255, 0.03)',
                          border: isSelected 
                            ? '1px solid var(--accent-purple)' 
                            : '1px solid var(--border-light)',
                          color: isSelected ? 'white' : 'var(--text-secondary)',
                          padding: '8px 14px',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          fontWeight: 500,
                          fontSize: '0.9rem',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          transition: 'all 0.2s'
                        }}
                      >
                        <code>{token}</code>
                        <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>
                          (ID {manifest.vocabData[token].token_id})
                        </span>
                      </button>
                    );
                  })}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-light)', paddingTop: '16px' }}>
                  <div style={{ display: 'flex', gap: '16px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    <div>
                      <span>CCE Iterations Limit:</span>{' '}
                      <input 
                        type="number" 
                        value={maxCceIter} 
                        onChange={e => setMaxCceIter(parseInt(e.target.value) || 10)}
                        style={{ width: '50px', background: 'transparent', border: '1px solid var(--border-light)', color: 'white', padding: '2px 4px', borderRadius: '4px', textAlign: 'center', marginLeft: '6px' }}
                      />
                    </div>
                    <div>
                      <span>Convergence Threshold (Epsilon):</span>{' '}
                      <input 
                        type="number" 
                        step="0.05"
                        value={epsilon} 
                        onChange={e => setEpsilon(parseFloat(e.target.value) || 0.1)}
                        style={{ width: '60px', background: 'transparent', border: '1px solid var(--border-light)', color: 'white', padding: '2px 4px', borderRadius: '4px', textAlign: 'center', marginLeft: '6px' }}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button 
                      className="btn-secondary"
                      onClick={() => setSelectedTokens([])}
                      style={{ padding: '6px 12px', fontSize: '0.85rem' }}
                    >
                      <RotateCcw size={14} /> Clear Sequence
                    </button>
                  </div>
                </div>
              </div>

              {/* Autocomplete Predictions Panel */}
              <div className="glass-panel" style={{ padding: '20px' }}>
                <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sparkles size={18} color="var(--accent-cyan)" /> Auto-Complete Output Projections
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                  The final state vector projects vocabulary tokens based on similarity to target slider definitions in <code>output_rules.json</code>.
                </p>

                {autocompleteSuggestions.length > 0 ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '10px' }}>
                    {autocompleteSuggestions.map(s => {
                      const isHigh = s.similarity >= 80;
                      const isMedium = s.similarity >= 50 && s.similarity < 80;
                      let simColor = 'var(--text-muted)';
                      if (isHigh) simColor = 'var(--accent-emerald)';
                      else if (isMedium) simColor = 'var(--accent-amber)';

                      return (
                        <div key={s.token} className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '6px', borderLeft: `3px solid ${simColor}`, background: 'rgba(30, 41, 59, 0.2)' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <strong style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', color: 'white' }}>
                              {s.token}
                            </strong>
                            <span style={{ fontSize: '0.8rem', fontWeight: 'bold', color: simColor }}>
                              {s.similarity}% Match
                            </span>
                          </div>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.3' }}>
                            {s.description}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Choose sequence tokens to analyze autocomplete suggestions.
                  </div>
                )}
              </div>

              {/* Execution Trace Stepper */}
              <div className="glass-panel" style={{ padding: '20px', flex: 1 }}>
                <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={18} color="var(--accent-blue)" /> Layer-by-Layer Execution Trace
                </h3>

                {simulationResult ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {simulationResult.trace.map((step, idx) => {
                      const isActive = idx === selectedStepIdx;
                      const isCce = step.type === 'CCE';
                      const layerMetadata = manifest.architecture?.layers?.find((l: any) => l.layer_id === step.layer);
                      
                      return (
                        <div 
                          key={idx}
                          onClick={() => setSelectedStepIdx(idx)}
                          className="glass-card"
                          style={{
                            cursor: 'pointer',
                            borderColor: isActive ? 'var(--accent-purple)' : 'var(--border-light)',
                            background: isActive ? 'rgba(139, 92, 246, 0.05)' : 'var(--bg-panel)',
                            padding: '12px 16px',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center'
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                            <div style={{
                              width: '28px',
                              height: '28px',
                              borderRadius: '50%',
                              background: isCce ? 'rgba(244, 63, 94, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                              color: isCce ? 'var(--accent-rose)' : 'var(--accent-blue)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontWeight: 700,
                              fontSize: '0.85rem',
                              border: `1px solid ${isCce ? 'rgba(244, 63, 94, 0.3)' : 'rgba(59, 130, 246, 0.3)'}`
                            }}>
                              {step.layer}
                            </div>

                            <div>
                              <div style={{ fontWeight: 600, fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                Layer {step.layer}: {layerMetadata?.name || 'Unknown Layer'}
                                <span className={`badge ${isCce ? 'badge-rose' : 'badge-cyan'}`}>
                                  {layerMetadata?.type || step.type}
                                </span>
                              </div>
                              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px', fontStyle: 'italic' }}>
                                {layerMetadata?.mechanism}
                              </div>
                              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px', lineHeight: '1.4' }}>
                                {layerMetadata?.description}
                              </div>
                              {isCce && (
                                <div style={{ fontSize: '0.8rem', color: 'var(--accent-rose)', marginTop: '4px' }}>
                                  Cognitive loop run: {step.cce_info?.iterations} steps. {step.cce_info?.converged ? 'Resolved below threshold.' : 'Halted at iterations ceiling.'}
                                </div>
                              )}
                              {isActive && layerMetadata?.prompt && (
                                <div style={{ marginTop: '8px', padding: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px', fontSize: '0.8rem', borderLeft: '3px solid var(--accent-purple)' }}>
                                  <strong>Prompt:</strong> <em>{layerMetadata.prompt}</em>
                                </div>
                              )}
                              {isActive && layerMetadata?.matrix_example && (
                                <div style={{ marginTop: '4px', padding: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px', fontSize: '0.75rem', fontFamily: 'monospace', whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
                                  {layerMetadata.matrix_example}
                                </div>
                              )}
                            </div>
                          </div>

                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            {/* Rule counter badge */}
                            {isCce ? (
                              <span className="badge badge-purple">
                                CCE Active
                              </span>
                            ) : (
                              <span className="badge badge-purple" style={{ opacity: step.active_rules?.some(ar => ar.length > 0) ? 1 : 0.4 }}>
                                {step.active_rules?.flat().length || 0} Rules Fired
                              </span>
                            )}
                            <ArrowRight size={16} color={isActive ? 'var(--accent-purple)' : 'var(--text-muted)'} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                    Select at least one input token to run the CNVM execution simulation.
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Register State Visualizer Grid */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              
              <div className="glass-panel" style={{ padding: '20px', height: '100%', display: 'flex', flexDirection: 'column' }}>
                <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Cpu size={18} color="var(--accent-cyan)" /> State Register Grid Visualizer
                </h3>

                <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '16px' }}>
                  Hidden state vector at the selected trace step. Hover cells to read coordinates.
                </p>

                    {simulationResult && simulationResult.trace[selectedStepIdx]?.type === 'CCE' && (
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-light)', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>CCE Iteration Loop:</span>
                      <strong style={{ color: 'var(--accent-purple)' }}>
                        Step {selectedCceIter} / {simulationResult.trace[selectedStepIdx].cce_info!.history.length - 1}
                      </strong>
                    </div>
                    <input 
                      type="range" 
                      min="0"
                      max={simulationResult.trace[selectedStepIdx].cce_info!.history.length - 1}
                      value={selectedCceIter}
                      onChange={e => setSelectedCceIter(parseInt(e.target.value) || 0)}
                      style={{ width: '100%', accentColor: 'var(--accent-purple)' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginTop: '4px', color: 'var(--text-secondary)' }}>
                      <span>Start (Attn)</span>
                      <span>Conflict: {simulationResult.trace[selectedStepIdx].cce_info!.history[selectedCceIter]?.conflict_value.toFixed(4)}</span>
                      <span>Converged</span>
                    </div>
                  </div>
                )}

                {/* Token Selector */}
                {simulationResult && selectedTokens.length > 1 && (
                  <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', background: 'rgba(255,255,255,0.02)', padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--border-light)', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Select Token to Visualize:</span>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      {selectedTokens.map((t, idx) => (
                        <button
                          key={idx}
                          onClick={() => setVisualizedTokenIdx(idx)}
                          className="tab-btn"
                          style={{
                            padding: '4px 8px',
                            fontSize: '0.75rem',
                            background: visualizedTokenIdx === idx ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                            border: `1px solid ${visualizedTokenIdx === idx ? 'var(--accent-purple)' : 'var(--border-light)'}`,
                            color: visualizedTokenIdx === idx ? 'white' : 'var(--text-secondary)'
                          }}
                        >
                          <code>{t}</code>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Grid Visualizer */}
                {activeStateVector ? (
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    
                    {/* Execution Explanation Panel */}
                    {(() => {
                      const currentStepRules = simulationResult?.trace[selectedStepIdx]?.active_rules?.[visualizedTokenIdx] || [];
                      if (currentStepRules.length === 0) return null;
                      
                      return (
                        <div style={{ background: 'rgba(139, 92, 246, 0.05)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                          <h4 style={{ fontSize: '0.85rem', color: 'var(--accent-purple)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <Info size={14} /> Execution Explanation (What Happened)
                          </h4>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                            {currentStepRules.map(ar => {
                              const ruleInfo = ruleMap[ar.rule_id];
                              if (!ruleInfo) return null;
                              return (
                                <div key={ar.rule_id} style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                  <strong style={{ color: 'var(--text-primary)' }}>{ruleInfo.name}</strong> fired: 
                                  triggered by <code style={{ color: 'var(--accent-emerald)' }}>{ruleInfo.trigger_slider_name}</code>, 
                                  updated <code style={{ color: 'var(--accent-cyan)' }}>{ruleInfo.result_slider_name}</code>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })()}

                    {/* Render grid of active sliders as chips */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '8px', maxHeight: '420px', overflowY: 'auto', paddingRight: '4px' }}>
                      {Object.entries(manifest.tsrMap)
                        .filter(([_, [start]]) => {
                          const val = (activeStateVector[visualizedTokenIdx] || activeStateVector[0] || [])[start];
                          return Math.abs(val || 0) > 1e-4 && Math.abs((val || 0) + 2) > 1e-4;
                        })
                        .map(([domain, [start]]) => {
                          const val = (activeStateVector[visualizedTokenIdx] || activeStateVector[0] || [])[start];

                          const isMetadata = domain === "PROVENANCE_ID" || domain === "EVIDENCE_TRACE";
                          
                          let bgStyle = 'rgba(30, 41, 59, 0.4)';
                          let borderColor = 'rgba(255, 255, 255, 0.1)';
                          let textColor = 'white';
                          
                          if (isMetadata) {
                            borderColor = 'var(--accent-purple)';
                            textColor = 'var(--accent-purple)';
                          } else if (val > 0) {
                            borderColor = 'var(--accent-emerald)';
                            textColor = 'var(--accent-emerald)';
                          } else {
                            borderColor = 'var(--accent-rose)';
                            textColor = 'var(--accent-rose)';
                          }
                          
                          const sliderName = manifest.slidersConfig[domain]?.name || domain.split('::').pop() || domain;

                          return (
                            <div
                              key={domain}
                              title={getCellTooltip(domain, 0, val)}
                              onMouseEnter={() => setHoveredIdx(start)}
                              onMouseLeave={() => setHoveredIdx(null)}
                              className="glass-card"
                              style={{
                                display: 'flex',
                                flexDirection: 'column',
                                background: hoveredIdx === start ? 'rgba(255, 255, 255, 0.08)' : bgStyle,
                                borderLeft: `3px solid ${borderColor}`,
                                padding: '8px 12px',
                                cursor: 'crosshair',
                                transition: 'background 0.1s ease',
                                gap: '4px'
                              }}
                            >
                              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                {domain}
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <strong style={{ fontSize: '0.85rem', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                                  {sliderName}
                                </strong>
                                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: textColor, marginLeft: '8px' }}>
                                  {isMetadata ? val.toFixed(0) : (val > 0 ? '+' : '') + val.toFixed(2)}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                        
                      {Object.entries(manifest.tsrMap).every(([_, [start]]) => Math.abs((activeStateVector[visualizedTokenIdx] || activeStateVector[0] || [])[start] || 0) <= 1e-4) && (
                        <div style={{ padding: '20px', gridColumn: '1 / -1', textAlign: 'center', color: 'var(--text-muted)' }}>
                          No active registers (all values are 0) for this token at this step.
                        </div>
                      )}
                    </div>

                    {/* Selected Active cell details panel */}
                    <div className="glass-card" style={{ marginTop: 'auto', background: 'rgba(30, 41, 59, 0.2)', padding: '12px', minHeight: '90px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      {hoveredDetail ? (
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                              {hoveredDetail.name}
                            </span>
                            <span className="badge badge-purple" style={{ fontSize: '0.65rem' }}>
                              Index {hoveredIdx}
                            </span>
                          </div>
                          <p style={{ fontSize: '0.75rem', color: 'var(--text-primary)', lineHeight: 1.3 }}>
                            {hoveredDetail.description}
                          </p>
                          <div style={{ fontSize: '0.8rem', fontWeight: 'bold', marginTop: '4px', display: 'flex', gap: '8px' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>Activation:</span>
                            <span style={{ 
                              color: activeStateVector[visualizedTokenIdx][hoveredIdx!] > 0 
                                ? 'var(--accent-emerald)' 
                                : activeStateVector[visualizedTokenIdx][hoveredIdx!] < 0 ? 'var(--accent-rose)' : 'white' 
                            }}>
                              {activeStateVector[visualizedTokenIdx][hoveredIdx!].toFixed(6)}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                          <Info size={16} />
                          Hover over any active register chip to inspect its mathematical definition and current state activation.
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No state vector active. Run simulation first.
                  </div>
                )}
            </div>
          </div>
        </div>
      )}

        {/* TAB 5: EXECUTION GRAPH */}
        {activeTab === 'graph' && (
          <div className="glass-panel" style={{ padding: '20px', height: 'calc(100vh - 180px)', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} color="var(--accent-purple)" /> Execution Graph
            </h3>
            {simulationResult && selectedTokens.length > 0 ? (
              <ExecutionGraph 
                checkpoint={checkpoint}
                manifest={manifest}
                tokens={selectedTokens}
                cceLayers={cceLayers}
                maxCceIter={maxCceIter}
                epsilon={epsilon}
              />
            ) : (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                Please run a simulation in the Simulator tab first to generate the execution graph.
              </div>
            )}
          </div>
        )}

        {/* TAB 2: JSON SCHEMA EDITOR */}
        {activeTab === 'editor' && (
          <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '20px' }}>
            
            {/* Left Column: File Tree list */}
            <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <h4 style={{ color: 'var(--text-primary)', marginBottom: '8px', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Manifest JSON Files
              </h4>

              {Object.keys(fileTexts).map(fileName => {
                const isActive = selectedFile === fileName;
                return (
                  <button
                    key={fileName}
                    onClick={() => setSelectedFile(fileName)}
                    style={{
                      background: isActive ? 'rgba(139, 92, 246, 0.1)' : 'transparent',
                      border: isActive ? '1px solid rgba(139, 92, 246, 0.2)' : '1px solid transparent',
                      color: isActive ? 'var(--accent-purple)' : 'var(--text-secondary)',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      textAlign: 'left',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.8rem',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      transition: 'all 0.2s'
                    }}
                  >
                    <BookOpen size={14} />
                    {fileName}
                  </button>
                );
              })}
            </div>

            {/* Right Column: Code editor area */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              
              <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Settings size={18} color="var(--accent-purple)" /> Manifest Schema Compiler Interface
                    </h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                      Currently editing: <strong style={{ color: 'white' }}>manifest/{selectedFile}</strong>
                    </p>
                  </div>

                  <button 
                    className="btn-primary"
                    onClick={handleCompile}
                  >
                    <Save size={16} /> Compile Schema changes
                  </button>
                </div>

                {compileError && (
                  <div className="glass-card" style={{ border: '1px solid var(--accent-rose)', background: 'rgba(244, 63, 94, 0.05)', color: 'var(--accent-rose)', display: 'flex', gap: '12px', alignItems: 'center', padding: '12px' }}>
                    <AlertTriangle size={20} />
                    <div style={{ fontSize: '0.85rem', fontFamily: 'var(--font-mono)' }}>
                      <strong>Compilation Failed:</strong> {compileError}
                    </div>
                  </div>
                )}

                <textarea
                  className="code-textarea"
                  value={fileTexts[selectedFile] || ""}
                  onChange={e => {
                    setFileTexts({
                      ...fileTexts,
                      [selectedFile]: e.target.value
                    });
                  }}
                  spellCheck="false"
                />

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  <Info size={14} />
                  Editing files in this visual sandbox allows testing new attention bridges or SERG rules dynamically without modifying the Python backend files. Click 'Compile Schema' to rebuild the running execution checkpoint.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: WEIGHTS INSPECTOR */}
        {activeTab === 'weights' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Embedding Visualizer */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Table size={18} color="var(--accent-cyan)" /> Vocab Token Embeddings Matrix
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                Inspect the compiled coordinates and values inside the embedding matrix. Values are inherited from the parent token class and clamped to [-2.0, 2.0].
              </p>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-light)', color: 'var(--text-secondary)' }}>
                      <th style={{ padding: '10px' }}>Token name</th>
                      <th style={{ padding: '10px' }}>ID</th>
                      <th style={{ padding: '10px' }}>Inherited class</th>
                      <th style={{ padding: '10px' }}>Active Sliders</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(manifest.vocabData).map(([token, info]) => {
                      const id = checkpoint.token_to_id[token];
                      const vector = checkpoint.embedding[id] || [];

                      // Identify coordinates with non-zero values
                      const activeSliders: string[] = [];
                      for (const [name, [start]] of Object.entries(manifest.tsrMap)) {
                        if (name !== 'RESERVED' && Math.abs(vector[start]) > 1e-4) {
                          activeSliders.push(`${name}: ${vector[start].toFixed(2)}`);
                        }
                      }

                      return (
                        <tr key={token} style={{ borderBottom: '1px solid var(--border-light)' }}>
                          <td style={{ padding: '10px', fontWeight: 'bold' }}><code>{token}</code></td>
                          <td style={{ padding: '10px' }}>{id}</td>
                          <td style={{ padding: '10px', color: 'var(--accent-cyan)' }}>{info.inherits || 'None'}</td>
                          <td style={{ padding: '10px' }}>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                              {activeSliders.map(s => (
                                <span key={s} className="badge badge-purple" style={{ fontSize: '0.7rem' }}>{s}</span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* FFN Rules and Attention Bridges compilation inspector */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              
              {/* Attention Routing Fabric Bridge Inspector */}
              <div className="glass-panel" style={{ padding: '20px' }}>
                <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Search size={18} color="var(--accent-blue)" /> Sparse Inter-Domain Bridges (SIDBs)
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                  Visualizes active attention bridges compiled between separate domain coordinates. Identity diagonals exist for isolation, and bridges punch holes to route signals under attention.
                </p>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {Object.entries(manifest.routingData).map(([src, config]) => (
                    <div key={src} className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'white' }}>
                        Source: <code>{src}</code>
                      </div>
                      <div style={{ paddingLeft: '12px', borderLeft: '2px solid var(--accent-blue)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        {config.bridges && Object.entries(config.bridges).map(([tgt, weight]) => (
                          <div key={tgt} style={{ fontSize: '0.8rem', display: 'flex', justifyContent: 'space-between' }}>
                            <span>Target: <code>{tgt}</code></span>
                            <span style={{ fontWeight: 'bold', color: 'var(--accent-cyan)' }}>
                              Weight: {weight.toFixed(3)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* SERG Rules Inspector */}
              <div className="glass-panel" style={{ padding: '20px' }}>
                <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={18} color="var(--accent-amber)" /> Sparse Executable Rule Graph (SERG)
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                  FFN rules compiled into key-value projection weights (W_in and W_out). Rule_id acts as metadata and writes back trace details on trigger.
                </p>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '350px', overflowY: 'auto' }}>
                  {Object.entries(manifest.rulesData).map(([name, rule]) => (
                    <div key={name} className="glass-card" style={{ fontSize: '0.8rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <strong style={{ color: 'white' }}>{name}</strong>
                        <span className="badge badge-emerald" style={{ fontSize: '0.65rem' }}>ID {rule.rule_id}</span>
                      </div>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginBottom: '6px' }}>{rule.intent_description}</p>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.75rem' }}>
                        <div>Trigger: <code>{rule.trigger_slider_name}</code> ({rule.gate_in_weight})</div>
                        <div>Output: <code>{rule.result_slider_name}</code> ({rule.gate_out_weight})</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

            </div>

          </div>
        )}

        {/* TAB 4: KNOWLEDGE BASE */}
        {activeTab === 'knowledge' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* Training Knowledge Sentences */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <BookOpen size={18} color="var(--accent-purple)" /> Training Knowledge — Raw Sentences
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                These are the raw knowledge sentences the model was trained on. Each sentence maps a token sequence to a known answer.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {[
                  { domain: 'History of Canada', color: 'var(--accent-purple)', sentences: [
                    { q: 'What is the capital of Canada?', tokens: ['canada', 'capital'], answer: 'ottawa' },
                    { q: 'Who was the first Prime Minister of Canada?', tokens: ['first', 'prime_minister', 'canada'], answer: 'john_a_macdonald' },
                    { q: 'What year was Confederation?', tokens: ['year', 'confederation'], answer: '1867' },
                    { q: 'What is Canada\'s national symbol?', tokens: ['national', 'symbol', 'canada'], answer: 'maple_leaf' },
                    { q: 'What is the capital of Canada first?', tokens: ['capital', 'canada', 'first'], answer: 'ottawa' },
                    { q: 'Who was the first PM of Canada that year?', tokens: ['first', 'prime_minister', 'canada', 'year'], answer: 'john_a_macdonald' },
                    { q: 'What is Canada\'s national symbol?', tokens: ['canada', 'national', 'symbol'], answer: 'maple_leaf' },
                  ]},
                  { domain: 'Cooking & Culinary Science', color: 'var(--accent-amber)', sentences: [
                    { q: 'What temperature should you bake a souffle at?', tokens: ['bake', 'temperature', 'souffle'], answer: '375f' },
                    { q: 'What makes bread rise?', tokens: ['bread', 'rise'], answer: 'yeast' },
                    { q: 'What temperature is a medium rare steak?', tokens: ['steak', 'medium_rare'], answer: '140f' },
                    { q: 'What temperature for a medium rare steak?', tokens: ['steak', 'temperature', 'medium_rare'], answer: '140f' },
                    { q: 'A souffle must bake at temperature to rise', tokens: ['souffle', 'bake', 'temperature', 'rise'], answer: '375f' },
                    { q: 'Bread needs baking agent to rise', tokens: ['bread', 'bake', 'rise'], answer: 'yeast' },
                    { q: 'Bake a steak to medium rare', tokens: ['steak', 'bake', 'medium_rare'], answer: '140f' },
                  ]},
                  { domain: 'Car Cleaning & Surface Care', color: 'var(--accent-emerald)', sentences: [
                    { q: 'What is the safest cloth for car detailing?', tokens: ['safest', 'cloth'], answer: 'microfiber' },
                    { q: 'What protects car paint from UV damage?', tokens: ['protects', 'uv'], answer: 'carnauba_wax' },
                    { q: 'What cleans leather car seats?', tokens: ['cleans', 'leather', 'seats'], answer: 'ph_neutral_cleaner' },
                    { q: 'What safely cleans leather seats?', tokens: ['cleans', 'leather', 'seats', 'safest'], answer: 'microfiber' },
                    { q: 'Safest cloth for leather seats', tokens: ['safest', 'cloth', 'leather', 'seats'], answer: 'microfiber' },
                    { q: 'What cleans and protects leather?', tokens: ['cleans', 'protects', 'leather'], answer: 'ph_neutral_cleaner' },
                  ]},
                ].map(group => (
                  <div key={group.domain} style={{ marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                      <div style={{ width: '4px', height: '20px', borderRadius: '2px', background: group.color }} />
                      <h4 style={{ color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600 }}>{group.domain}</h4>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', paddingLeft: '12px' }}>
                      {group.sentences.map((s, i) => (
                        <div key={i} style={{ 
                          display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '12px', alignItems: 'center',
                          padding: '10px 14px', borderRadius: '8px',
                          background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-light)',
                          fontSize: '0.85rem'
                        }}>
                          <div style={{ color: 'var(--text-primary)' }}>
                            <span style={{ color: 'var(--text-muted)', marginRight: '8px' }}>Q:</span>
                            {s.q}
                          </div>
                          <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                            {s.tokens.map((t, j) => (
                              <code key={j} style={{
                                padding: '2px 6px', borderRadius: '4px', fontSize: '0.75rem',
                                background: 'rgba(139, 92, 246, 0.15)', color: 'var(--accent-purple)',
                                border: '1px solid rgba(139, 92, 246, 0.2)'
                              }}>{t}</code>
                            ))}
                          </div>
                          <div style={{ 
                            padding: '3px 10px', borderRadius: '6px', fontWeight: 600,
                            background: `rgba(16, 185, 129, 0.15)`, color: 'var(--accent-emerald)',
                            border: '1px solid rgba(16, 185, 129, 0.2)', fontSize: '0.8rem', whiteSpace: 'nowrap'
                          }}>
                            → {s.answer}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Vocabulary Definitions */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={18} color="var(--accent-cyan)" /> Vocabulary Definitions
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                All tokens in the compiled vocabulary with their concept descriptions and slider configurations.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '500px', overflowY: 'auto' }}>
                {Object.entries(manifest.vocabData).sort(([a], [b]) => a.localeCompare(b)).map(([token, data]) => (
                  <div key={token} style={{
                    display: 'grid', gridTemplateColumns: '150px 1fr auto', gap: '12px', alignItems: 'start',
                    padding: '10px 14px', borderRadius: '8px',
                    background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-light)',
                    fontSize: '0.85rem'
                  }}>
                    <div>
                      <code style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>{token}</code>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>ID: {data.token_id}</div>
                    </div>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', lineHeight: 1.4 }}>
                      {data.concept_description}
                    </div>
                    <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                      {Object.entries(data.sliders || data.overrides || {}).map(([slider, config]) => (
                        <span key={slider} style={{
                          padding: '2px 6px', borderRadius: '4px', fontSize: '0.7rem',
                          background: 'rgba(251, 191, 36, 0.1)', color: 'var(--accent-amber)',
                          border: '1px solid rgba(251, 191, 36, 0.15)', whiteSpace: 'nowrap'
                        }}>
                          {slider}: {(config as any).value}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Output Rules Reference */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ArrowRight size={18} color="var(--accent-emerald)" /> Output Rules — Target Slider Definitions
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                These define the expected final-state slider weights for each output token. The auto-complete engine matches the simulation\'s final state against these targets.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {Object.entries(manifest.outputData).sort(([a], [b]) => a.localeCompare(b)).map(([token, rule]) => (
                  <div key={token} style={{
                    padding: '12px 16px', borderRadius: '8px',
                    background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-light)'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <code style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--accent-emerald)' }}>{token}</code>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {rule.token_id}</span>
                      </div>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{rule.intent_description}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                      {Object.entries(rule.target_sliders || {}).map(([slider, config]) => {
                        const w = (config as any).weight;
                        const isPos = w > 0;
                        return (
                          <span key={slider} style={{
                            padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', fontFamily: 'var(--font-mono)',
                            background: isPos ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
                            color: isPos ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                            border: `1px solid ${isPos ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)'}`,
                            whiteSpace: 'nowrap'
                          }}>
                            {slider}: {w}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* SERG Layer Rules Reference */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <h3 style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu size={18} color="var(--accent-amber)" /> SERG Layer Rules
              </h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
                All compiled rules across all layers, showing trigger/result slider mappings and gate weights.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '500px', overflowY: 'auto' }}>
                {Object.entries(manifest.rulesData)
                  .sort(([, a], [, b]) => (a.layer_index ?? 0) - (b.layer_index ?? 0))
                  .map(([ruleName, rule]) => (
                  <div key={ruleName} style={{
                    display: 'grid', gridTemplateColumns: '60px 200px 1fr', gap: '12px', alignItems: 'center',
                    padding: '10px 14px', borderRadius: '8px',
                    background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-light)',
                    fontSize: '0.8rem'
                  }}>
                    <span style={{
                      padding: '3px 8px', borderRadius: '6px', textAlign: 'center', fontWeight: 600,
                      background: 'rgba(251, 191, 36, 0.1)', color: 'var(--accent-amber)',
                      border: '1px solid rgba(251, 191, 36, 0.15)', fontSize: '0.75rem'
                    }}>L{rule.layer_index}</span>
                    <code style={{ color: 'var(--text-primary)', fontSize: '0.8rem' }}>{ruleName}</code>
                    <div style={{ display: 'flex', gap: '12px', color: 'var(--text-secondary)' }}>
                      <span>⇨ <code>{rule.trigger_slider_name}</code> ({rule.gate_in_weight})</span>
                      <span>→ <code>{rule.result_slider_name}</code> ({rule.gate_out_weight})</span>
                      <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>{rule.intent_description}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

      </main>
    </div>
  );
}
