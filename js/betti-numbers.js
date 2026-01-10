// Betti Numbers Computation Module
// Computes the Betti numbers of the order complex of the poset of neuron activation regions

const BettiNumbers = {
    // Cache for computed results
    lastPoset: null,
    lastResult: null,
    isComputing: false,

    // Toggle state
    showBettiNumbers: false,
    lastPosetState: null,  // Track the last poset to detect changes

    /**
     * Main function to compute Betti numbers
     * Called when the "Compute Betti Numbers" button is clicked
     */
    compute() {
        if (this.isComputing) {
            return;
        }

        const displayEl = document.getElementById('betti-numbers-display');
        if (!displayEl) return;

        this.isComputing = true;
        displayEl.innerHTML = '<div style="color: #666; font-style: italic;">Computing...</div>';

        // Use setTimeout to allow UI to update
        setTimeout(() => {
            try {
                const result = this.computeBettiNumbers();
                this.displayResults(result);
            } catch (error) {
                console.error('Error computing Betti numbers:', error);
                displayEl.innerHTML = `<div style="color: #e74c3c;">Error: ${error.message}</div>`;
            } finally {
                this.isComputing = false;
            }
        }, 10);
    },

    /**
     * Compute Betti numbers of the order complex of the poset
     * @returns {Object} Result object with Betti numbers and diagnostics
     */
    computeBettiNumbers() {
        // Get the nerve complex data (poset)
        const nerveData = BinaryDisplay.computeUniversalNerveComplex();

        if (!nerveData.success) {
            throw new Error(nerveData.error || 'Failed to compute nerve complex');
        }

        const { simplices, edges } = nerveData;

        if (simplices.length === 0) {
            return {
                bettiNumbers: [],
                error: 'No simplices in poset'
            };
        }

        // Build the order complex from the poset
        const orderComplex = this.buildOrderComplex({ simplices, edges });

        // Compute homology using chain complexes
        const bettiNumbers = this.computeHomology(orderComplex);

        // Cache the result
        this.lastPoset = { simplices, edges };
        this.lastResult = {
            bettiNumbers,
            orderComplex,
            posetSize: simplices.length,
            numEdges: edges.length
        };

        return this.lastResult;
    },

    /**
     * Build the order complex from the poset
     * The order complex Δ(P) has:
     * - Vertices: elements of the poset (simplices)
     * - k-simplices: chains (totally ordered sets) of k+1 elements
     *
     * @param {Object} poset - Object with simplices and edges
     * @returns {Object} Order complex organized by dimension
     */
    buildOrderComplex(poset) {
        const { simplices, edges } = poset;

        // Build adjacency list for the Hasse diagram
        const adjacency = new Map(); // Maps simplex index -> list of larger simplex indices
        const reverseAdj = new Map(); // Maps simplex index -> list of smaller simplex indices

        for (let i = 0; i < simplices.length; i++) {
            adjacency.set(i, []);
            reverseAdj.set(i, []);
        }

        for (const edge of edges) {
            adjacency.get(edge.from).push(edge.to);
            reverseAdj.get(edge.to).push(edge.from);
        }

        // Find all chains (totally ordered subsets) in the poset
        // A chain is a sequence s0 < s1 < ... < sk where each is contained in the next
        const chains = new Map(); // Map from dimension to list of chains

        // Dimension 0: Each element of the poset is a 0-simplex in the order complex
        chains.set(0, simplices.map((_, idx) => [idx]));

        // Find all maximal chains using DFS
        const allChains = [];

        // DFS to find all chains starting from each vertex
        const findChainsFrom = (current, path) => {
            const currentPath = [...path, current];
            allChains.push(currentPath);

            // Explore all larger elements
            for (const next of adjacency.get(current)) {
                findChainsFrom(next, currentPath);
            }
        };

        // Start from each element
        for (let i = 0; i < simplices.length; i++) {
            findChainsFrom(i, []);
        }

        // Group chains by dimension (k-chain has k+1 elements, so dimension k)
        for (const chain of allChains) {
            const dim = chain.length - 1;
            if (!chains.has(dim)) {
                chains.set(dim, []);
            }

            // Convert chain to canonical form (sorted)
            const canonicalChain = [...chain].sort((a, b) => a - b);

            // Check if this chain already exists (avoid duplicates)
            const chainKey = canonicalChain.join(',');
            const existingChains = chains.get(dim);
            const isDuplicate = existingChains.some(c => c.join(',') === chainKey);

            if (!isDuplicate) {
                existingChains.push(canonicalChain);
            }
        }

        // Find maximum dimension
        let maxDim = 0;
        for (const dim of chains.keys()) {
            if (dim > maxDim) maxDim = dim;
        }

        console.log('Order complex constructed:');
        for (let d = 0; d <= maxDim; d++) {
            const count = chains.get(d)?.length || 0;
            console.log(`  Dimension ${d}: ${count} chains`);
        }

        return {
            chains,
            maxDim,
            adjacency,
            reverseAdj,
            posetSize: simplices.length
        };
    },

    /**
     * Compute homology groups and Betti numbers
     * Uses chain complex and boundary operators
     *
     * @param {Object} orderComplex - The order complex structure
     * @returns {Array<number>} Betti numbers β_k for each dimension k
     */
    computeHomology(orderComplex) {
        const { chains, maxDim } = orderComplex;
        const bettiNumbers = [];

        // For each dimension, compute the Betti number
        for (let k = 0; k <= maxDim; k++) {
            const chainsK = chains.get(k) || [];
            const chainsKMinus1 = chains.get(k - 1) || [];
            const chainsKPlus1 = chains.get(k + 1) || [];

            if (chainsK.length === 0) {
                bettiNumbers[k] = 0;
                continue;
            }

            // Compute boundary matrices
            const boundaryK = this.computeBoundaryMatrix(chainsK, chainsKMinus1);
            const boundaryKPlus1 = this.computeBoundaryMatrix(chainsKPlus1, chainsK);

            // Compute ranks using Gaussian elimination (mod 2)
            const rankK = this.computeMatrixRank(boundaryK);
            const rankKPlus1 = this.computeMatrixRank(boundaryKPlus1);

            // Betti number: β_k = dim(C_k) - rank(∂_k) - rank(∂_{k+1})
            // This is the rank of the k-th homology group H_k = ker(∂_k) / im(∂_{k+1})
            const betti = chainsK.length - rankK - rankKPlus1;
            bettiNumbers[k] = Math.max(0, betti); // Ensure non-negative

            console.log(`β_${k}: dim=${chainsK.length}, rank(∂_${k})=${rankK}, rank(∂_${k+1})=${rankKPlus1} → β_${k}=${bettiNumbers[k]}`);
        }

        return bettiNumbers;
    },

    /**
     * Compute the boundary matrix ∂_k: C_k -> C_{k-1}
     * Entry [i][j] is the coefficient of chain_j in the boundary of chain_i
     *
     * Working over Z/2Z (mod 2), so all non-zero coefficients are 1
     *
     * @param {Array<Array<number>>} chainsK - k-dimensional chains (domain)
     * @param {Array<Array<number>>} chainsKMinus1 - (k-1)-dimensional chains (codomain)
     * @returns {Array<Array<number>>} Boundary matrix (rows = chainsK, cols = chainsKMinus1)
     */
    computeBoundaryMatrix(chainsK, chainsKMinus1) {
        if (!chainsK || chainsK.length === 0 || !chainsKMinus1 || chainsKMinus1.length === 0) {
            return [];
        }

        const rows = chainsK.length;
        const cols = chainsKMinus1.length;
        const matrix = Array(rows).fill(null).map(() => Array(cols).fill(0));

        // For each k-chain, compute its boundary as a sum of (k-1)-chains
        for (let i = 0; i < chainsK.length; i++) {
            const chain = chainsK[i];

            // The boundary of a chain [s0, s1, ..., sk] is the alternating sum:
            // ∂([s0,...,sk]) = Σ (-1)^i [s0,...,ŝi,...,sk]
            // where ŝi means omit si
            //
            // In Z/2Z, (-1)^i = 1, so we just add (mod 2)

            for (let j = 0; j < chain.length; j++) {
                // Create the face by removing element j
                const face = [...chain.slice(0, j), ...chain.slice(j + 1)];

                // Find this face in chainsKMinus1
                const faceKey = face.join(',');
                const faceIdx = chainsKMinus1.findIndex(c => c.join(',') === faceKey);

                if (faceIdx !== -1) {
                    // In Z/2Z: add 1 (mod 2)
                    matrix[i][faceIdx] = (matrix[i][faceIdx] + 1) % 2;
                }
            }
        }

        return matrix;
    },

    /**
     * Compute the rank of a matrix over Z/2Z using Gaussian elimination
     *
     * @param {Array<Array<number>>} matrix - Matrix with entries in {0, 1}
     * @returns {number} Rank of the matrix
     */
    computeMatrixRank(matrix) {
        if (!matrix || matrix.length === 0 || matrix[0].length === 0) {
            return 0;
        }

        // Create a copy to avoid modifying the original
        const m = matrix.map(row => [...row]);
        const rows = m.length;
        const cols = m[0].length;

        let rank = 0;
        let currentRow = 0;

        // Gaussian elimination to row echelon form
        for (let col = 0; col < cols && currentRow < rows; col++) {
            // Find pivot (first non-zero entry in this column at or below currentRow)
            let pivotRow = -1;
            for (let row = currentRow; row < rows; row++) {
                if (m[row][col] === 1) {
                    pivotRow = row;
                    break;
                }
            }

            // No pivot found in this column
            if (pivotRow === -1) {
                continue;
            }

            // Swap rows if needed
            if (pivotRow !== currentRow) {
                [m[currentRow], m[pivotRow]] = [m[pivotRow], m[currentRow]];
            }

            // Eliminate all other 1s in this column (mod 2)
            for (let row = 0; row < rows; row++) {
                if (row !== currentRow && m[row][col] === 1) {
                    // Add currentRow to row (mod 2)
                    for (let c = 0; c < cols; c++) {
                        m[row][c] = (m[row][c] + m[currentRow][c]) % 2;
                    }
                }
            }

            rank++;
            currentRow++;
        }

        return rank;
    },

    /**
     * Display the computed Betti numbers in the UI
     * @param {Object} result - Result from computeBettiNumbers
     */
    displayResults(result) {
        const displayEl = document.getElementById('betti-numbers-display');
        if (!displayEl) return;

        if (result.error) {
            displayEl.innerHTML = `<div style="color: #e74c3c;">${result.error}</div>`;
            return;
        }

        const { bettiNumbers, posetSize, numEdges } = result;

        let html = '<div style="margin-bottom: 8px;">';
        html += '<div style="font-weight: bold; color: #2c3e50; margin-bottom: 5px;">Betti Numbers of Order Complex:</div>';

        // Display each Betti number with interpretation
        const interpretations = [
            'connected components',
            '1-dimensional holes (loops)',
            '2-dimensional holes (voids)',
            '3-dimensional holes',
            '4-dimensional holes',
            '5-dimensional holes'
        ];

        if (bettiNumbers.length === 0) {
            html += '<div style="color: #666; font-style: italic;">No Betti numbers computed</div>';
        } else {
            // Use MathJax if available for nice formatting
            for (let i = 0; i < bettiNumbers.length; i++) {
                const interpretation = i < interpretations.length ?
                    ` <span style="color: #666; font-size: 11px;">(${interpretations[i]})</span>` : '';

                // Format with subscript using HTML entities or MathJax
                const betaSymbol = `β<sub>${i}</sub>`;

                html += `<div style="margin: 3px 0; font-family: 'Courier New', monospace;">`;
                html += `${betaSymbol} = <strong style="color: #2980b9;">${bettiNumbers[i]}</strong>${interpretation}`;
                html += `</div>`;
            }
        }

        html += '</div>';

        // Add summary statistics
        html += '<div style="border-top: 1px solid #e0e0e0; margin-top: 8px; padding-top: 8px; font-size: 11px; color: #666;">';
        html += `Poset: ${posetSize} elements, ${numEdges} edges<br>`;
        html += `Order complex: up to dimension ${bettiNumbers.length - 1}`;
        html += '</div>';

        displayEl.innerHTML = html;

        // Trigger MathJax rendering if available
        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
            MathJax.typesetPromise([displayEl]).catch(err => console.log('MathJax error:', err));
        }
    },

    /**
     * Clear the cached results
     */
    clearCache() {
        this.lastPoset = null;
        this.lastResult = null;
    },

    /**
     * Check if the poset has changed since last computation
     * @param {Object} newPoset - Object with simplices and edges arrays
     * @returns {boolean} True if poset has changed
     */
    hasPosetChanged(newPoset) {
        if (!this.lastPosetState || !newPoset) return true;

        // Compare the poset structure
        const oldSimplices = this.lastPosetState.simplices;
        const newSimplices = newPoset.simplices;

        // Check if number of simplices changed
        if (oldSimplices.length !== newSimplices.length) return true;

        // Check if edges changed
        const oldEdges = JSON.stringify(this.lastPosetState.edges.sort());
        const newEdges = JSON.stringify(newPoset.edges.sort());
        if (oldEdges !== newEdges) return true;

        // Check if simplices themselves changed
        const oldSimplexSet = new Set(oldSimplices.map(s =>
            JSON.stringify([...s].sort())
        ));
        const newSimplexSet = new Set(newSimplices.map(s =>
            JSON.stringify([...s].sort())
        ));

        if (oldSimplexSet.size !== newSimplexSet.size) return true;

        for (const simplex of newSimplexSet) {
            if (!oldSimplexSet.has(simplex)) return true;
        }

        return false;
    },

    /**
     * Update Betti numbers if needed (when toggle is on and poset has changed)
     * Called automatically by nerve complex rendering
     * @param {Object} poset - Object with simplices and edges arrays
     */
    updateIfNeeded(poset) {
        if (!this.showBettiNumbers) return;
        if (!poset || !poset.simplices) return;

        // Check if poset has changed
        if (this.hasPosetChanged(poset)) {
            this.computeAndDisplay(poset);
            // Deep copy the poset state for future comparison
            this.lastPosetState = {
                simplices: JSON.parse(JSON.stringify(poset.simplices)),
                edges: JSON.parse(JSON.stringify(poset.edges))
            };
        }
    },

    /**
     * Compute and display Betti numbers
     * @param {Object} posetParam - Optional poset object to use instead of computing
     */
    computeAndDisplay(posetParam = null) {
        const display = document.getElementById('betti-numbers-display');
        if (!display) return;

        // Show loading state
        display.innerHTML = '<div style="color: #666; font-style: italic;">Computing...</div>';

        // Use setTimeout to allow UI to update
        setTimeout(() => {
            try {
                // Get poset from parameter or from BinaryDisplay
                let poset = posetParam;
                if (!poset && typeof BinaryDisplay !== 'undefined') {
                    const result = BinaryDisplay.computeUniversalNerveComplex();
                    if (!result.success) {
                        display.innerHTML = `<div style="color: #e74c3c;">${result.error || result.message}</div>`;
                        return;
                    }
                    poset = { simplices: result.simplices, edges: result.edges };
                }

                if (!poset || !poset.simplices) {
                    display.innerHTML = '<div style="color: #e74c3c;">No poset data available</div>';
                    return;
                }

                // Build the order complex from the poset
                const orderComplex = this.buildOrderComplex(poset);

                // Compute homology using chain complexes
                const bettiNumbers = this.computeHomology(orderComplex);

                // Cache the result
                this.lastPoset = poset;
                this.lastResult = {
                    bettiNumbers,
                    orderComplex,
                    posetSize: poset.simplices.length,
                    numEdges: poset.edges.length
                };

                // Display results
                this.displayResults(this.lastResult);
            } catch (error) {
                console.error('Error computing Betti numbers:', error);
                display.innerHTML = `<div style="color: #e74c3c;">Error: ${error.message}</div>`;
            }
        }, 10);
    }
};

// Global function for toggle handler
function toggleBettiNumbers() {
    const checkbox = document.getElementById('betti-numbers-toggle');
    BettiNumbers.showBettiNumbers = checkbox.checked;

    if (checkbox.checked) {
        // Compute and display Betti numbers
        BettiNumbers.computeAndDisplay();
    } else {
        // Hide Betti numbers display
        const display = document.getElementById('betti-numbers-display');
        if (display) {
            display.innerHTML = '';
        }
    }
}

// Legacy global function for backward compatibility
function computeBettiNumbers() {
    BettiNumbers.compute();
}
