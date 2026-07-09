import React, { useMemo } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { CNVMRuntime } from '../cnvm/runtime';

const CustomEdge = ({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  data,
}: any) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />
      {data?.label && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
            }}
            className="nodrag nopan"
          >
            {data.label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

const edgeTypes = { customEdge: CustomEdge };

export function ExecutionGraph({ 
  checkpoint, 
  manifest, 
  tokens, 
  cceLayers, 
  maxCceIter, 
  epsilon 
}: { 
  checkpoint: any;
  manifest: any;
  tokens: string[];
  cceLayers: number[];
  maxCceIter: number;
  epsilon: number;
}) {
  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    if (!checkpoint || tokens.length === 0) return { nodes: [], edges: [] };
    
    const runtime = new CNVMRuntime(checkpoint, manifest.tsrMap, manifest.dim);
    const fullSim = runtime.runForward(tokens, undefined, cceLayers, maxCceIter, epsilon);
    const trace = fullSim.trace;
    
    const states: number[][][] = [];
    for (let i = 0; i <= trace.length; i++) {
      states.push(runtime.runForward(tokens, i, cceLayers, maxCceIter, epsilon).finalState);
    }
    
    // Build rule map
    const ruleMap: Record<number, any> = {};
    for (const [name, rule] of Object.entries(manifest.rulesData)) {
      if ((rule as any).rule_id !== undefined) {
        ruleMap[(rule as any).rule_id] = { name, ...(rule as any) };
      }
    }

    const newNodes: any[] = [];
    const newEdges: any[] = [];
    
    let yOffset = 0;
    const xSpacing = 500;
    
    // Create initial nodes for each token
    tokens.forEach((token, tIdx) => {
      newNodes.push({
        id: `t${tIdx}_layer0`,
        position: { x: tIdx * xSpacing, y: yOffset },
        data: { 
          label: (
            <div style={{ padding: '12px', minWidth: '180px' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Input Token</div>
              <strong style={{ fontSize: '1.2rem', color: 'var(--accent-cyan)' }}>{token}</strong>
            </div>
          ) 
        },
        style: {
          background: 'rgba(30, 41, 59, 0.9)',
          color: 'white',
          border: '1px solid var(--accent-cyan)',
          borderRadius: '8px',
        }
      });
    });
    
    yOffset += 350;
    
    let lastActiveNodeIdByToken = tokens.map((_, i) => `t${i}_layer0`);

    trace.forEach((step, stepIdx) => {
      const layerMeta = manifest.architecture.layers.find((l: any) => l.layer_id === step.layer);
      const prevState = states[stepIdx];
      const currState = states[stepIdx + 1];
      
      let stepHasChanges = false;
      const stepNodesY = yOffset;

      tokens.forEach((_, tIdx) => {
        // Compute diff for this token
        const prevVec = prevState[tIdx];
        const currVec = currState[tIdx];
        
        // Find fired rules FIRST to know targeted domains
        const activeRules = step.active_rules?.[tIdx] || [];
        const firedRuleNames = activeRules.map((ar: any) => ruleMap[ar.rule_id]?.name || `Rule ${ar.rule_id}`);
        const targetedDomains = new Set(activeRules.map((ar: any) => ruleMap[ar.rule_id]?.result_slider_name));
        
        const changes: { domain: string, oldVal: number, newVal: number, diff: number }[] = [];
        Object.entries(manifest.tsrMap as Record<string, [number, number]>).forEach(([domain, [start]]) => {
          if (domain === "META::PROVENANCE" || domain === "META::EVIDENCE") return;
          
          // For standard SERG layers, ONLY show changes for sliders explicitly targeted by the rules
          // This hides the massive amount of noise caused by layer normalization and attention diffusion.
          if (step.type !== 'CCE' && !targetedDomains.has(domain)) return;
          
          const oldVal = prevVec[start];
          const newVal = currVec[start];
          const diffThreshold = step.type === 'CCE' ? 0.1 : 1e-4; // Higher threshold for CCE to hide gradient noise
          
          if (Math.abs(newVal - oldVal) > diffThreshold) {
            const oldUnused = Math.abs(oldVal) <= 1e-4 || Math.abs(oldVal + 2) <= 1e-4;
            const newUnused = Math.abs(newVal) <= 1e-4 || Math.abs(newVal + 2) <= 1e-4;
            
            if (!(oldUnused && newUnused)) {
              changes.push({ domain, oldVal, newVal, diff: newVal - oldVal });
            }
          }
        });
        
        if (changes.length > 0 || firedRuleNames.length > 0) {
          stepHasChanges = true;
          const nodeId = `t${tIdx}_layer${step.layer}`;
          
          newNodes.push({
            id: nodeId,
            position: { x: tIdx * xSpacing, y: stepNodesY },
            data: {
              label: (
                <div style={{ padding: '16px', minWidth: '360px', maxWidth: '420px', textAlign: 'left' }}>
                  <div style={{ fontSize: '0.9rem', color: 'var(--accent-purple)', fontWeight: 'bold' }}>
                    Layer {step.layer} ({step.type})
                  </div>
                  <div style={{ fontSize: '1.1rem', color: 'white', marginBottom: '8px' }}>
                    {layerMeta?.name || 'Unknown Layer'}
                  </div>
                  
                  {firedRuleNames.length > 0 && (
                    <div style={{ fontSize: '0.9rem', color: 'var(--accent-emerald)', marginBottom: '8px' }}>
                      Rules Fired: {firedRuleNames.join(', ')}
                    </div>
                  )}
                  
                </div>
              )
            },
            style: {
              background: 'rgba(30, 41, 59, 0.95)',
              color: 'white',
              border: `1px solid ${step.type === 'CCE' ? 'var(--accent-rose)' : 'var(--accent-purple)'}`,
              borderRadius: '8px',
            }
          });
          
          const edgeLabel = changes.length > 0 ? (
            <div style={{ background: 'rgba(15, 23, 42, 0.95)', border: '1px solid rgba(255,255,255,0.2)', padding: '10px', borderRadius: '8px', minWidth: '220px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', textAlign: 'center', fontWeight: 'bold' }}>Weight Changes</div>
              {changes.map((c, i) => (
                <div key={i} style={{ fontSize: '0.75rem', display: 'flex', justifyContent: 'space-between', fontFamily: 'monospace' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{c.domain}</span>
                  <span style={{ color: c.diff > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                    {c.diff > 0 ? '+' : ''}{c.diff.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ background: 'rgba(30, 41, 59, 0.9)', padding: '4px 8px', borderRadius: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
              Forward State
            </div>
          );

          newEdges.push({
            id: `e_${lastActiveNodeIdByToken[tIdx]}_${nodeId}`,
            source: lastActiveNodeIdByToken[tIdx],
            target: nodeId,
            type: 'customEdge',
            animated: true,
            data: { label: edgeLabel },
            style: { stroke: 'var(--accent-purple)', strokeWidth: 2 },
            markerEnd: { type: MarkerType.ArrowClosed, color: 'var(--accent-purple)' },
          });
          
          lastActiveNodeIdByToken[tIdx] = nodeId;
        }
      });
      
      if (step.attn_weights) {
        tokens.forEach((_, tIdx) => {
          tokens.forEach((_, sIdx) => {
            if (tIdx === sIdx) return;
            const weight = step.attn_weights![tIdx][sIdx];
            if (weight > 0.3) {
              const sourceNodeId = lastActiveNodeIdByToken[sIdx];
              const targetNodeId = lastActiveNodeIdByToken[tIdx];
              if (sourceNodeId && targetNodeId && sourceNodeId !== targetNodeId) {
                newEdges.push({
                  id: `attn_${sourceNodeId}_${targetNodeId}_l${step.layer}`,
                  source: sourceNodeId,
                  target: targetNodeId,
                  animated: true,
                  label: `${(weight * 100).toFixed(0)}%`,
                  style: { stroke: 'var(--accent-cyan)', strokeDasharray: '5,5', strokeWidth: 1.5 },
                  labelStyle: { fill: 'var(--accent-cyan)', fontSize: '11px', fontWeight: 'bold' },
                  labelBgStyle: { fill: 'rgba(30, 41, 59, 0.8)' },
                  markerEnd: { type: MarkerType.ArrowClosed, color: 'var(--accent-cyan)' },
                });
              }
            }
          });
        });
      }
      
      if (stepHasChanges) {
        yOffset += 450;
      }
    });

    return { nodes: newNodes, edges: newEdges };
  }, [checkpoint, manifest, tokens, cceLayers, maxCceIter, epsilon]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update when deps change
  React.useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  return (
    <div style={{ width: '100%', height: '100%', minHeight: '600px', background: 'var(--bg-primary)', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-light)' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        panOnScroll={true}
        fitView
        colorMode="dark"
      >
        <Background gap={16} color="rgba(255,255,255,0.05)" />
        <Controls />
        <MiniMap 
          nodeColor={(n) => {
            if (n.id.includes('layer0')) return 'var(--accent-cyan)';
            return 'var(--accent-purple)';
          }} 
          maskColor="rgba(15, 23, 42, 0.7)"
        />
      </ReactFlow>
    </div>
  );
}
