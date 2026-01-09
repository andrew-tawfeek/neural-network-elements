// Binary Display Module
const BinaryDisplay = {
    // Configuration: Canvas size when loss graph is visible (change this value to adjust size)
    COMPACT_CANVAS_SIZE: 300,  // ← MODIFY THIS VALUE to change size when loss graph is active
    NORMAL_CANVAS_SIZE: 400,   // Normal size when no loss graph is present
    INPUT_2D_CANVAS_SIZE: 200, // ← MODIFY THIS VALUE to change size specifically for 2D input networks
    update() {
        const c = document.getElementById('binary-states');
        c.innerHTML = '';
        if (!window.net || !window.net.activations || window.net.activations.length === 0) return;
        
        let fullVector = [];
        
        // Show ONLY hidden layers (skip input and output)
        for (let l = 1; l < window.net.activations.length - 1; l++) {
            const div = document.createElement('div');
            div.className = 'binary-layer';
            
            const layerName = `Hidden ${l}`;
            
            const binaryVec = window.net.activations[l].map(neuron => neuron[0] > 0 ? 1 : 0);
            fullVector = fullVector.concat(binaryVec);
            div.innerHTML = `<div class="binary-title">${layerName}:</div><div class="binary-vector">[${binaryVec.join(', ')}]</div>`;
            c.appendChild(div);
        }
        
        const fullDiv = document.createElement('div');
        fullDiv.className = 'full-binary';
        
        // Check if there are highlighted neurons from the DecisionBoundary module
        let highlightedNeuronIndices = new Set();
        if (typeof DecisionBoundary !== 'undefined') {
            if (DecisionBoundary.highlightedNeurons && DecisionBoundary.highlightedNeurons.size > 0) {
                highlightedNeuronIndices = DecisionBoundary.highlightedNeurons;
            } else if (DecisionBoundary.highlightedNeuron !== null) {
                // Backward compatibility
                highlightedNeuronIndices.add(DecisionBoundary.highlightedNeuron);
            }
        }
        
        // Create the full vector display with highlighting
        let fullVectorHtml = '[';
        for (let i = 0; i < fullVector.length; i++) {
            if (highlightedNeuronIndices.has(i)) {
                // Make the highlighted neurons bold
                fullVectorHtml += `<strong style="background-color: #f39c12; color: white; padding: 1px 3px; border-radius: 2px;">${fullVector[i]}</strong>`;
            } else {
                fullVectorHtml += fullVector[i];
            }
            if (i < fullVector.length - 1) {
                fullVectorHtml += ', ';
            }
        }
        fullVectorHtml += ']';
        
        fullDiv.innerHTML = `<div class="full-binary-title">Full Network State:</div><div class="full-binary-vector">${fullVectorHtml}</div>`;
        c.appendChild(fullDiv);
        
        // Add neighboring states section
        this.addNeighboringStates(c, fullVector);
        
        // Add Graph Laplacian Spectrum section
        this.addSpectrumSection(c);
        
        // Update plot visibility
        this.updatePlotVisibility();
        
        // Store the current loss tracking status for future updates
        this.lastLossTrackingStatus = (typeof LossGraph !== 'undefined' && LossGraph.lossHistory && LossGraph.lossHistory.length > 0) ||
                                    (typeof window.data !== 'undefined' && window.data && window.data.length > 0) ||
                                    document.getElementById('error')?.innerHTML.includes('Loss:');
    },

    addNeighboringStates(container, currentState) {
        const neighborsDiv = document.createElement('div');
        neighborsDiv.className = 'neighboring-states';
        neighborsDiv.style.marginTop = '15px';
        neighborsDiv.style.paddingTop = '15px';
        neighborsDiv.style.borderTop = '2px solid #e0e0e0';
        
        // Find all neighboring states (Hamming distance = 1)
        const neighbors = this.findNeighboringStates(currentState);
        
        let neighborsHtml = `
            <div style="font-weight: bold; color: #2c3e50; margin-bottom: 6px; font-size:14px;">
                Neighboring States:
            </div>
        `;
        
        if (neighbors.length === 0) {
            neighborsHtml += `<div style="font-style: italic; color: #666;">No neighboring regions found</div>`;
        } else {
            const isFrom2D = window.net.architecture[0] === 2 && window.currentDualGraph;
            neighborsHtml += `<div style="margin-bottom: 5px; font-size: 12px; color: #666;">Found ${neighbors.length} neighboring region(s):</div>`;
            
            neighbors.forEach((neighbor, index) => {
                const diffPos = this.findDifference(currentState, neighbor.state);
                let neighborDescription = '';
                
                if (isFrom2D && neighbor.regionId !== undefined) {
                    // Neighbor from dual graph (2D input case)
                    neighborDescription = `
                        <div style="margin: 5px 0; padding: 5px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">
                            <div style="font-weight: 500;">Region ${neighbor.regionId}: [${neighbor.state.join(', ')}]</div>
                            <div style="color: #666; font-size: 11px;">Differs at neuron ${diffPos} (${currentState[diffPos]} → ${neighbor.state[diffPos]})</div>
                        </div>
                    `;
                } else if (neighbor.inputIndex !== undefined) {
                    // Neighbor from input variation (non-2D input case)
                    neighborDescription = `
                        <div style="margin: 5px 0; padding: 5px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">
                            <div style="font-weight: 500;">Neighbor ${index + 1}: [${neighbor.state.join(', ')}]</div>
                            <div style="color: #666; font-size: 11px;">
                                Reached by varying input ${neighbor.inputIndex + 1} by ${neighbor.inputVariation > 0 ? '+' : ''}${neighbor.inputVariation}
                            </div>
                            <div style="color: #666; font-size: 11px;">
                                Test inputs: [${neighbor.testInputs.map(v => v.toFixed(3)).join(', ')}]
                            </div>
                            <div style="color: #666; font-size: 11px;">Differs at neuron ${diffPos} (${currentState[diffPos]} → ${neighbor.state[diffPos]})</div>
                        </div>
                    `;
                } else {
                    // Fallback
                    neighborDescription = `
                        <div style="margin: 5px 0; padding: 5px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">
                            <div style="font-weight: 500;">Neighbor ${index + 1}: [${neighbor.state.join(', ')}]</div>
                            <div style="color: #666; font-size: 11px;">Differs at neuron ${diffPos} (${currentState[diffPos]} → ${neighbor.state[diffPos]})</div>
                        </div>
                    `;
                }
                
                neighborsHtml += neighborDescription;
            });
        }
        
        neighborsDiv.innerHTML = neighborsHtml;
        container.appendChild(neighborsDiv);
    },

    findNeighboringStates(currentState) {
        // If we have a complete dual graph, use it for finding neighbors
        if (window.currentDualGraph && window.currentDualGraph.nodes && window.currentDualGraph.edges && window.currentDualGraph.nodes.length > 1) {
            return this.findNeighborsFromDualGraph(currentState);
        }
        
        // For 2D inputs, use the decision boundary dual graph if available
        if (window.net.architecture[0] === 2 && window.currentDualGraph && window.currentDualGraph.nodes && window.currentDualGraph.edges) {
            return this.findNeighborsFromDualGraph(currentState);
        }
        
        // For all other cases, compute neighbors by varying input values
        return this.findNeighborsByInputVariation(currentState);
    },

    findNeighborsFromDualGraph(currentState) {
        const neighbors = [];
        const currentStateStr = currentState.join(',');
        
        // Find the current region in the dual graph
        let currentRegionId = -1;
        for (const node of window.currentDualGraph.nodes) {
            if (node.state === currentStateStr) {
                currentRegionId = node.id;
                break;
            }
        }
        
        if (currentRegionId === -1) {
            return []; // Current state not found in dual graph
        }
        
        // Find all edges connected to the current region
        const connectedRegionIds = new Set();
        for (const edgeStr of window.currentDualGraph.edges) {
            const [id1, id2] = edgeStr.split(',').map(Number);
            if (id1 === currentRegionId) {
                connectedRegionIds.add(id2);
            } else if (id2 === currentRegionId) {
                connectedRegionIds.add(id1);
            }
        }
        
        // Get the states of connected regions
        for (const regionId of connectedRegionIds) {
            const neighborNode = window.currentDualGraph.nodes.find(node => node.id === regionId);
            if (neighborNode && neighborNode.state) {
                const neighborState = neighborNode.state.split(',').map(Number);
                neighbors.push({
                    state: neighborState,
                    regionId: regionId
                });
            }
        }
        
        return neighbors;
    },

    findNeighborsByInputVariation(currentState) {
        if (!window.net) return [];
        
        const neighbors = [];
        const inputSize = window.net.architecture[0];
        
        // Get current input values from the UI
        const currentInputs = [];
        for (let i = 0; i < inputSize; i++) {
            const inputElement = document.getElementById(`input-${i}`);
            if (inputElement) {
                currentInputs.push(parseFloat(inputElement.value) || 0);
            }
        }
        
        if (currentInputs.length !== inputSize) {
            return []; // Unable to get input values
        }
        
        // Try small variations in each input dimension
        const variations = [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1];
        
        for (let inputIdx = 0; inputIdx < inputSize; inputIdx++) {
            for (const variation of variations) {
                const testInputs = [...currentInputs];
                testInputs[inputIdx] += variation;
                
                // Forward pass with test inputs
                const testOutput = window.net.forward(testInputs);
                
                // Get network state from hidden layers only using getBinaryState method
                const testState = window.net.getBinaryState();
                
                // Check if this state is different from current state and has Hamming distance = 1
                const hammingDistance = this.calculateHammingDistance(currentState, testState);
                if (hammingDistance === 1) {
                    // Check if we already have this neighbor
                    const stateStr = testState.join(',');
                    const exists = neighbors.some(n => n.state.join(',') === stateStr);
                    
                    if (!exists) {
                        neighbors.push({
                            state: testState,
                            inputIndex: inputIdx,
                            inputVariation: variation,
                            testInputs: [...testInputs]
                        });
                    }
                }
            }
        }
        
        // Restore original input state
        window.net.forward(currentInputs);
        
        return neighbors;
    },

    calculateHammingDistance(state1, state2) {
        if (state1.length !== state2.length) return Infinity;
        
        let distance = 0;
        for (let i = 0; i < state1.length; i++) {
            if (state1[i] !== state2[i]) {
                distance++;
            }
        }
        return distance;
    },

    findDifference(state1, state2) {
        for (let i = 0; i < state1.length; i++) {
            if (state1[i] !== state2[i]) {
                return i;
            }
        }
        return -1;
    },

    addSpectrumSection(container) {
        const spectrumDiv = document.createElement('div');
        spectrumDiv.id = 'spectrum-section';
        spectrumDiv.style.marginTop = '15px';
        spectrumDiv.style.paddingTop = '15px';
        spectrumDiv.style.borderTop = '2px solid #e0e0e0';
        spectrumDiv.innerHTML = `
            <div style="font-weight: bold; color: #2c3e50; margin-bottom: 6px; font-size:14px;">
                Graph Laplacian Spectrum:
            </div>
            <div id="eigenvalues-display" style="font-family: monospace; font-size: 13px; color: #333; margin-bottom: 10px;"></div>
            <div style="display: flex; gap: 5px;">
                <button onclick="computeSpectrum()" style="font-size: 12px; padding: 4px 8px;">Compute Spectrum</button>
            </div>
        `;
        container.appendChild(spectrumDiv);
    },

    /**
     * Compute the nerve complex from the dual graph
     * The nerve complex consists of all cliques in the dual graph
     * where vertices (activation regions) that share an edge have overlapping regions
     */
    computeNerveComplex() {
        if (!window.currentDualGraph || !window.currentDualGraph.nodes || window.currentDualGraph.nodes.length === 0) {
            return {
                success: false,
                error: 'No dual graph available. Please ensure 2D plots are visible.'
            };
        }

        const { nodes, edges } = window.currentDualGraph;
        const n = nodes.length;

        // Build adjacency list for efficient neighbor lookup
        const adjacencyList = new Map();
        for (let i = 0; i < n; i++) {
            adjacencyList.set(i, new Set());
        }

        edges.forEach(edgeStr => {
            const [id1, id2] = edgeStr.split(',').map(Number);
            adjacencyList.get(id1).add(id2);
            adjacencyList.get(id2).add(id1);
        });

        // Find all maximal cliques using Bron-Kerbosch algorithm
        const maximalCliques = this.findMaximalCliques(adjacencyList, n);

        // Build the poset (Hasse diagram) from the cliques
        const poset = this.buildPoset(maximalCliques);

        return {
            success: true,
            cliques: maximalCliques,
            poset: poset,
            numNodes: n
        };
    },

    /**
     * Bron-Kerbosch algorithm to find all maximal cliques
     * Returns array of cliques, where each clique is a Set of vertex indices
     */
    findMaximalCliques(adjacencyList, numVertices) {
        const maximalCliques = [];

        // Convert adjacency list Map to a more convenient format
        const neighbors = new Map();
        for (let i = 0; i < numVertices; i++) {
            neighbors.set(i, adjacencyList.get(i) || new Set());
        }

        // Bron-Kerbosch with pivoting
        const bronKerbosch = (R, P, X) => {
            if (P.size === 0 && X.size === 0) {
                // R is a maximal clique
                if (R.size > 0) {
                    maximalCliques.push(new Set(R));
                }
                return;
            }

            // Choose pivot from P ∪ X with most neighbors in P
            const union = new Set([...P, ...X]);
            let pivot = null;
            let maxNeighbors = -1;

            for (const v of union) {
                const neighborsInP = [...neighbors.get(v)].filter(n => P.has(n)).length;
                if (neighborsInP > maxNeighbors) {
                    maxNeighbors = neighborsInP;
                    pivot = v;
                }
            }

            // Iterate over P \ neighbors(pivot)
            const pivotNeighbors = pivot !== null ? neighbors.get(pivot) : new Set();
            const candidates = [...P].filter(v => !pivotNeighbors.has(v));

            for (const v of candidates) {
                const vNeighbors = neighbors.get(v);
                bronKerbosch(
                    new Set([...R, v]),
                    new Set([...P].filter(n => vNeighbors.has(n))),
                    new Set([...X].filter(n => vNeighbors.has(n)))
                );
                P.delete(v);
                X.add(v);
            }
        };

        // Initialize with all vertices in P
        const P = new Set();
        for (let i = 0; i < numVertices; i++) {
            P.add(i);
        }

        bronKerbosch(new Set(), P, new Set());

        // Add singleton cliques (0-simplices) for vertices not in any larger clique
        const verticesInCliques = new Set();
        maximalCliques.forEach(clique => {
            clique.forEach(v => verticesInCliques.add(v));
        });

        for (let i = 0; i < numVertices; i++) {
            if (!verticesInCliques.has(i)) {
                maximalCliques.push(new Set([i]));
            }
        }

        return maximalCliques;
    },

    /**
     * Build poset structure from maximal cliques
     * Returns object with nodes (simplices) and edges (containment relations)
     */
    buildPoset(maximalCliques) {
        // Generate all simplices (faces) from maximal cliques
        const allSimplices = new Set();

        maximalCliques.forEach(clique => {
            // Generate all subsets (faces) of this clique
            const faces = this.generateSubsets(Array.from(clique));
            faces.forEach(face => {
                if (face.length > 0) {
                    allSimplices.add(JSON.stringify(face.sort((a, b) => a - b)));
                }
            });
        });

        // Convert back to arrays and sort by dimension then lexicographically
        const simplices = Array.from(allSimplices)
            .map(s => JSON.parse(s))
            .sort((a, b) => {
                if (a.length !== b.length) return a.length - b.length;
                for (let i = 0; i < a.length; i++) {
                    if (a[i] !== b[i]) return a[i] - b[i];
                }
                return 0;
            });

        // Build containment relationships (edges in Hasse diagram)
        // An edge exists from simplex A to simplex B if A ⊂ B and dim(B) = dim(A) + 1
        const edges = [];

        for (let i = 0; i < simplices.length; i++) {
            for (let j = i + 1; j < simplices.length; j++) {
                const smaller = simplices[i];
                const larger = simplices[j];

                // Check if smaller is a proper subset of larger and dimensions differ by 1
                if (larger.length === smaller.length + 1 && this.isSubset(smaller, larger)) {
                    edges.push({ from: i, to: j });
                }
            }
        }

        return {
            simplices: simplices,
            edges: edges
        };
    },

    /**
     * Generate all subsets of an array
     */
    generateSubsets(arr) {
        const result = [];
        const n = arr.length;

        for (let i = 0; i < (1 << n); i++) {
            const subset = [];
            for (let j = 0; j < n; j++) {
                if (i & (1 << j)) {
                    subset.push(arr[j]);
                }
            }
            result.push(subset);
        }

        return result;
    },

    /**
     * Check if array a is a subset of array b
     */
    isSubset(a, b) {
        return a.every(item => b.includes(item));
    },

    /**
     * Map global neuron index to (layer, local_index) pair
     * Returns {layer: layer_index, neuron: neuron_index_in_layer, label: "Ln.m"}
     */
    getNeuronMapping(globalIndex) {
        if (!window.net || !window.net.architecture) {
            return { layer: 0, neuron: globalIndex, label: `${globalIndex}` };
        }

        let currentIndex = 0;
        // Iterate through hidden layers only (skip input layer 0 and output layer)
        for (let layer = 1; layer < window.net.architecture.length - 1; layer++) {
            const layerSize = window.net.architecture[layer];
            if (globalIndex < currentIndex + layerSize) {
                const localNeuron = globalIndex - currentIndex;
                return {
                    layer: layer,
                    neuron: localNeuron,
                    label: `${layer}_${localNeuron + 1}` // 1-indexed for display
                };
            }
            currentIndex += layerSize;
        }

        // Fallback
        return { layer: 0, neuron: globalIndex, label: `${globalIndex}` };
    },

    /**
     * Compute the universal nerve complex by sampling the input space
     * This is viewport-independent and explores all possible activation patterns
     */
    computeUniversalNerveComplex() {
        if (!window.net) {
            return {
                success: false,
                error: 'No neural network available.'
            };
        }

        const inputDim = window.net.architecture[0];
        let totalHiddenNeurons = 0;

        // Calculate total number of hidden neurons
        for (let i = 1; i < window.net.architecture.length - 1; i++) {
            totalHiddenNeurons += window.net.architecture[i];
        }

        if (totalHiddenNeurons === 0) {
            return {
                success: false,
                error: 'Network has no hidden layers.'
            };
        }

        // Store original network state
        const originalActivations = window.net.activations ?
            window.net.activations.map(layer => layer.map(a => [...a])) : null;

        // Track which sets of neurons can be simultaneously active
        // Key: sorted neuron indices as string, Value: true
        const coactivationSets = new Set();

        // Sample the input space
        let samplePoints;

        if (inputDim === 1) {
            // 1D: Sample along a line
            const numSamples = 100;
            const range = 20; // From -10 to 10
            samplePoints = [];
            for (let i = 0; i < numSamples; i++) {
                const x = -10 + (i / (numSamples - 1)) * range;
                samplePoints.push([x]);
            }
        } else if (inputDim === 2) {
            // 2D: Sample on a grid
            const gridSize = 50;
            const range = 20; // From -10 to 10
            samplePoints = [];
            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const x = -10 + (i / (gridSize - 1)) * range;
                    const y = -10 + (j / (gridSize - 1)) * range;
                    samplePoints.push([x, y]);
                }
            }
        } else {
            // Higher dimensions: Use random sampling combined with systematic sampling
            const numSamples = 1000;
            const range = 20;
            samplePoints = [];

            // Add some systematic samples (corners, center, axes)
            const systematicSamples = Math.pow(2, Math.min(inputDim, 4)); // Up to 16 corners for 4D
            for (let i = 0; i < systematicSamples; i++) {
                const point = [];
                for (let d = 0; d < inputDim; d++) {
                    point.push((i & (1 << d)) ? 10 : -10);
                }
                samplePoints.push(point);
            }

            // Add random samples
            for (let i = 0; i < numSamples - systematicSamples; i++) {
                const point = [];
                for (let d = 0; d < inputDim; d++) {
                    point.push(-10 + Math.random() * range);
                }
                samplePoints.push(point);
            }
        }

        // Sample each point and collect activation patterns
        const activationPatterns = new Set();

        for (const point of samplePoints) {
            window.net.forward(point);

            // Get the binary state (only hidden neurons)
            const binaryState = window.net.getBinaryState();
            const stateStr = binaryState.join(',');

            // Skip if we've seen this pattern before
            if (activationPatterns.has(stateStr)) {
                continue;
            }
            activationPatterns.add(stateStr);

            // Find all active neurons (indices where activation > 0)
            const activeNeurons = [];
            for (let i = 0; i < binaryState.length; i++) {
                if (binaryState[i] === 1) {
                    activeNeurons.push(i);
                }
            }

            // Add all non-empty subsets of active neurons to the coactivation sets
            // Each subset represents neurons that CAN be active together
            const subsets = this.generateSubsets(activeNeurons);
            for (const subset of subsets) {
                if (subset.length > 0) {
                    const key = subset.sort((a, b) => a - b).join(',');
                    coactivationSets.add(key);
                }
            }
        }

        // Restore original network state
        if (originalActivations) {
            window.net.activations = originalActivations;
        }

        // Convert coactivation sets to simplices
        const simplices = Array.from(coactivationSets)
            .map(str => str.split(',').map(Number))
            .sort((a, b) => {
                // Sort by dimension (length) first, then lexicographically
                if (a.length !== b.length) return a.length - b.length;
                for (let i = 0; i < a.length; i++) {
                    if (a[i] !== b[i]) return a[i] - b[i];
                }
                return 0;
            });

        // Build Hasse diagram edges (containment relations)
        const edges = [];
        for (let i = 0; i < simplices.length; i++) {
            for (let j = i + 1; j < simplices.length; j++) {
                const smaller = simplices[i];
                const larger = simplices[j];

                // Check if smaller is a proper subset of larger and dimensions differ by 1
                if (larger.length === smaller.length + 1 && this.isSubset(smaller, larger)) {
                    edges.push({ from: i, to: j });
                }
            }
        }

        console.log(`Universal nerve complex computed: ${simplices.length} simplices from ${activationPatterns.size} unique activation patterns`);

        return {
            success: true,
            simplices: simplices,
            edges: edges,
            numPatterns: activationPatterns.size,
            numSamples: samplePoints.length
        };
    },

    /**
     * Render the nerve complex as an SVG Hasse diagram
     */
    renderNerveComplex(nerveData) {
        const container = document.getElementById('nerve-complex-display');
        if (!container) return;

        const { simplices, edges } = nerveData.poset;

        if (simplices.length === 0) {
            container.innerHTML = '<div style="color: #666; font-style: italic;">No simplices found.</div>';
            return;
        }

        // Update info section
        const infoDiv = document.getElementById('nerve-complex-info');
        const dimensions = new Map();
        simplices.forEach(s => {
            const dim = s.length - 1;
            dimensions.set(dim, (dimensions.get(dim) || 0) + 1);
        });

        const dimInfo = Array.from(dimensions.entries())
            .sort((a, b) => a[0] - b[0])
            .map(([dim, count]) => `${count} ${dim}-simplex${count > 1 ? 'es' : ''}`)
            .join(', ');

        // Calculate total number of hidden neurons for proper layer count
        let totalHiddenNeurons = 0;
        if (window.net && window.net.architecture) {
            for (let i = 1; i < window.net.architecture.length - 1; i++) {
                totalHiddenNeurons += window.net.architecture[i];
            }
        }

        const maxPossibleDim = totalHiddenNeurons;
        infoDiv.textContent = `Found ${simplices.length} total simplices: ${dimInfo}. Max dimension: ${maxPossibleDim}`;

        // Layout parameters
        const nodeRadius = 20;
        const levelHeight = 80;
        const minNodeSpacing = 60;
        const padding = 40;

        // Group simplices by dimension - ensure all layers from 0 to maxPossibleDim exist
        const byDimension = new Map();
        for (let dim = 0; dim <= maxPossibleDim; dim++) {
            byDimension.set(dim, []);
        }

        simplices.forEach((simplex, idx) => {
            const dim = simplex.length - 1;
            if (dim <= maxPossibleDim) {
                byDimension.get(dim).push({ simplex, idx });
            }
        });

        const maxDim = maxPossibleDim;
        const maxWidth = Math.max(...Array.from(byDimension.values()).map(arr => arr.length), 1);

        // Calculate positions
        const positions = new Map();
        let yOffset = padding;

        for (let dim = 0; dim <= maxDim; dim++) {
            const nodesAtLevel = byDimension.get(dim) || [];
            const levelWidth = Math.max(nodesAtLevel.length * minNodeSpacing, 400);
            const xSpacing = levelWidth / (nodesAtLevel.length + 1);

            nodesAtLevel.forEach((item, i) => {
                positions.set(item.idx, {
                    x: padding + (i + 1) * xSpacing,
                    y: yOffset
                });
            });

            yOffset += levelHeight;
        }

        // Calculate SVG dimensions
        const svgWidth = Math.max(maxWidth * minNodeSpacing + 2 * padding, 500);
        const svgHeight = yOffset;

        // Create SVG
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', svgWidth);
        svg.setAttribute('height', svgHeight);
        svg.setAttribute('style', 'border: 1px solid #ddd; background: #fafafa; border-radius: 4px;');

        // Color scheme by dimension
        const colors = [
            '#3498db', // 0-simplices: blue
            '#2ecc71', // 1-simplices: green
            '#f39c12', // 2-simplices: orange
            '#e74c3c', // 3-simplices: red
            '#9b59b6', // 4-simplices: purple
            '#1abc9c', // 5-simplices: turquoise
        ];

        // Draw edges first (so they appear behind nodes)
        edges.forEach(edge => {
            const from = positions.get(edge.from);
            const to = positions.get(edge.to);

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', from.x);
            line.setAttribute('y1', from.y);
            line.setAttribute('x2', to.x);
            line.setAttribute('y2', to.y);
            line.setAttribute('stroke', '#999');
            line.setAttribute('stroke-width', '1.5');
            line.setAttribute('stroke-opacity', '0.6');
            svg.appendChild(line);
        });

        // Draw nodes
        simplices.forEach((simplex, idx) => {
            const pos = positions.get(idx);
            const dim = simplex.length - 1;
            const color = colors[dim % colors.length];

            // Create group for node and label
            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('data-simplex-idx', idx);

            // Circle
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', pos.x);
            circle.setAttribute('cy', pos.y);
            circle.setAttribute('r', nodeRadius);
            circle.setAttribute('fill', color);
            circle.setAttribute('stroke', '#333');
            circle.setAttribute('stroke-width', '2');
            circle.setAttribute('opacity', '0.85');
            circle.setAttribute('class', 'nerve-node');

            // Label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', pos.x);
            text.setAttribute('y', pos.y);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('dominant-baseline', 'middle');
            text.setAttribute('font-size', '11');
            text.setAttribute('font-weight', 'bold');
            text.setAttribute('fill', '#fff');
            text.setAttribute('pointer-events', 'none');
            text.textContent = simplex.join(',');

            // Tooltip title
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            const neuronText = simplex.length === 1 ? 'Neuron' : 'Neurons';
            title.textContent = `${dim}-simplex: ${neuronText} {${simplex.join(', ')}}\nDimension: ${dim}`;

            g.appendChild(circle);
            g.appendChild(text);
            g.appendChild(title);

            // Add hover effects
            g.addEventListener('mouseenter', () => {
                circle.setAttribute('opacity', '1');
                circle.setAttribute('stroke-width', '3');
            });

            g.addEventListener('mouseleave', () => {
                circle.setAttribute('opacity', '0.85');
                circle.setAttribute('stroke-width', '2');
            });

            svg.appendChild(g);
        });

        // Add legend
        const legendY = 20;
        const legendX = svgWidth - 150;

        const legendGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        const legendTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        legendTitle.setAttribute('x', legendX);
        legendTitle.setAttribute('y', legendY);
        legendTitle.setAttribute('font-size', '12');
        legendTitle.setAttribute('font-weight', 'bold');
        legendTitle.setAttribute('fill', '#333');
        legendTitle.textContent = 'Dimension:';
        legendGroup.appendChild(legendTitle);

        Array.from(dimensions.keys()).sort((a, b) => a - b).forEach((dim, i) => {
            const y = legendY + 20 + i * 20;

            const legendCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            legendCircle.setAttribute('cx', legendX);
            legendCircle.setAttribute('cy', y);
            legendCircle.setAttribute('r', '8');
            legendCircle.setAttribute('fill', colors[dim % colors.length]);
            legendCircle.setAttribute('stroke', '#333');
            legendCircle.setAttribute('stroke-width', '1');

            const legendText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            legendText.setAttribute('x', legendX + 15);
            legendText.setAttribute('y', y);
            legendText.setAttribute('font-size', '11');
            legendText.setAttribute('dominant-baseline', 'middle');
            legendText.setAttribute('fill', '#333');
            legendText.textContent = `${dim}-simplex`;

            legendGroup.appendChild(legendCircle);
            legendGroup.appendChild(legendText);
        });

        svg.appendChild(legendGroup);

        // Clear container and add SVG
        container.innerHTML = '';
        container.appendChild(svg);

        // Add CSS for hover effects
        const style = document.createElement('style');
        style.textContent = `
            .nerve-node {
                cursor: pointer;
                transition: opacity 0.2s, stroke-width 0.2s;
            }
        `;
        if (!document.getElementById('nerve-complex-styles')) {
            style.id = 'nerve-complex-styles';
            document.head.appendChild(style);
        }
    },

    updatePlotVisibility() {
        const plotContainer = document.getElementById('plot-container');
        const plot3dContainer = document.getElementById('3d-plot-container');
        
        // Always show the plot container if it exists (for 2D networks with decision boundaries)
        if (window.net.architecture[0] === 2 && window.net.architecture[window.net.architecture.length-1] === 1) {
            if (plotContainer && plotContainer.parentElement) {
                plotContainer.parentElement.style.display = 'block';
                const switch_el = document.getElementById('plot-type-switch');
                if (switch_el && switch_el.classList.contains('active')) {
                    plotContainer.style.display = 'none';
                    plot3dContainer.style.display = 'block';
                    if (typeof Visualization3D !== 'undefined') {
                        Visualization3D.update();
                    }
                } else {
                    plotContainer.style.display = 'block';
                    plot3dContainer.style.display = 'none';
                    if (typeof DecisionBoundary !== 'undefined') {
                        DecisionBoundary.update();
                    }
                }
            }
        } else if (plotContainer && plotContainer.parentElement) {
            // For non-2D networks, hide the decision boundary plots
            plotContainer.parentElement.style.display = 'none';
        }
    }
};

// Global function for backward compatibility
function updateBinaryDisplay() {
    BinaryDisplay.update();
}
