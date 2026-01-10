// Nerve Complex Toggle Module
// Handles toggling between Dual Graph and Nerve Complex visualizations
const NerveComplexToggle = {
    isNerveView: false,

    /**
     * Toggle between Dual Graph and Nerve Complex views
     */
    toggle() {
        const switchEl = document.getElementById('dual-nerve-switch');
        const label = document.getElementById('dual-nerve-label');
        const canvas = document.getElementById('dual-graph-canvas');
        const svgContainer = document.getElementById('nerve-complex-svg-container');
        const dualLegend = document.getElementById('dual-graph-legend');
        const nerveLegend = document.getElementById('nerve-complex-legend');
        const exportDiv = document.getElementById('dual-graph-export');
        const bettiSection = document.getElementById('betti-numbers-section');

        this.isNerveView = !this.isNerveView;

        if (this.isNerveView) {
            // Switch to Nerve Complex view
            switchEl.classList.add('active');
            label.textContent = 'Nerve Complex';
            canvas.style.display = 'none';
            svgContainer.style.display = 'block';
            dualLegend.style.display = 'none';
            nerveLegend.style.display = 'block';
            exportDiv.style.display = 'none';
            if (bettiSection) bettiSection.style.display = 'block';

            // Render the nerve complex
            this.renderNerveComplex();
        } else {
            // Switch to Dual Graph view
            switchEl.classList.remove('active');
            label.textContent = 'Dual Graph';
            canvas.style.display = 'block';
            svgContainer.style.display = 'none';
            dualLegend.style.display = 'block';
            nerveLegend.style.display = 'none';
            exportDiv.style.display = 'flex';
            if (bettiSection) bettiSection.style.display = 'none';
        }
    },

    /**
     * Render the nerve complex visualization in the SVG container
     */
    renderNerveComplex() {
        const container = document.getElementById('nerve-complex-svg-container');
        if (!container) return;

        // Compute universal nerve complex (viewport-independent)
        const nerveData = BinaryDisplay.computeUniversalNerveComplex();

        if (!nerveData.success) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: #666;">${nerveData.error}</div>`;
            return;
        }

        const { simplices, edges } = nerveData;

        if (simplices.length === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #666; font-style: italic;">No simplices found.</div>';
            return;
        }

        // Calculate total number of hidden neurons
        let totalHiddenNeurons = 0;
        if (window.net && window.net.architecture) {
            for (let i = 1; i < window.net.architecture.length - 1; i++) {
                totalHiddenNeurons += window.net.architecture[i];
            }
        }

        // Maximum cardinality is the total number of hidden neurons
        const maxPossibleCardinality = totalHiddenNeurons;

        // Layout parameters optimized for 300x300 canvas
        const nodeRadius = 15;
        const levelHeight = 35;
        const minNodeSpacing = 40;
        const padding = 30;
        const fontSize = 9;

        // Group simplices by CARDINALITY (size of set, not dimension)
        // Layer 0: Empty set (optional - we'll skip it for now as it's not meaningful)
        // Layer 1: Singletons {i}
        // Layer 2: Pairs {i,j}
        // Layer k: k-element sets
        const byCardinality = new Map();

        // Initialize all possible cardinality levels (1 to maxPossibleCardinality)
        for (let card = 1; card <= maxPossibleCardinality; card++) {
            byCardinality.set(card, []);
        }

        simplices.forEach((simplex, idx) => {
            const cardinality = simplex.length; // Cardinality is the size of the set
            if (cardinality > 0 && cardinality <= maxPossibleCardinality) {
                byCardinality.get(cardinality).push({ simplex, idx });
            }
        });

        // Find the actual maximum cardinality present in the data
        let maxCardinality = 1;
        for (let card = maxPossibleCardinality; card >= 1; card--) {
            if (byCardinality.get(card).length > 0) {
                maxCardinality = card;
                break;
            }
        }

        const maxWidth = Math.max(...Array.from(byCardinality.values()).map(arr => arr.length), 1);

        // Calculate positions
        // Layout: Empty set at top (layer 0), larger sets toward bottom
        const positions = new Map();
        let yOffset = padding;

        for (let card = 1; card <= maxCardinality; card++) {
            const nodesAtLevel = byCardinality.get(card) || [];

            if (nodesAtLevel.length === 0) {
                // Empty layer - still allocate space
                yOffset += levelHeight;
                continue;
            }

            const levelWidth = Math.max(nodesAtLevel.length * minNodeSpacing, 300 - 2 * padding);
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
        const svgWidth = Math.max(maxWidth * minNodeSpacing + 2 * padding, 300);
        const svgHeight = Math.max(yOffset, 300);

        // Create SVG
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', svgWidth);
        svg.setAttribute('height', svgHeight);
        svg.setAttribute('style', 'background: #fafafa;');

        // Color scheme by cardinality
        const colors = [
            '#999999', // Cardinality 0 (not used): gray
            '#3498db', // Cardinality 1 (singletons): blue
            '#2ecc71', // Cardinality 2 (pairs): green
            '#f39c12', // Cardinality 3 (triples): orange
            '#e74c3c', // Cardinality 4: red
            '#9b59b6', // Cardinality 5: purple
            '#1abc9c', // Cardinality 6: turquoise
            '#e67e22', // Cardinality 7: carrot
            '#34495e', // Cardinality 8+: dark gray
        ];

        // Draw edges first (so they appear behind nodes)
        edges.forEach(edge => {
            const from = positions.get(edge.from);
            const to = positions.get(edge.to);

            if (from && to) {
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', from.x);
                line.setAttribute('y1', from.y);
                line.setAttribute('x2', to.x);
                line.setAttribute('y2', to.y);
                line.setAttribute('stroke', '#999');
                line.setAttribute('stroke-width', '1');
                line.setAttribute('stroke-opacity', '0.4');
                svg.appendChild(line);
            }
        });

        // Draw cardinality labels on the left
        for (let card = 1; card <= maxCardinality; card++) {
            const nodesAtLevel = byCardinality.get(card) || [];
            if (nodesAtLevel.length > 0) {
                const firstNode = positions.get(nodesAtLevel[0].idx);
                if (firstNode) {
                    const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    labelText.setAttribute('x', 5);
                    labelText.setAttribute('y', firstNode.y);
                    labelText.setAttribute('font-size', '10');
                    labelText.setAttribute('font-weight', 'bold');
                    labelText.setAttribute('fill', '#666');
                    labelText.setAttribute('dominant-baseline', 'middle');
                    labelText.textContent = `|${card}|`; // Show cardinality as |n|
                    svg.appendChild(labelText);
                }
            }
        }

        // Draw nodes
        simplices.forEach((simplex, idx) => {
            const pos = positions.get(idx);
            if (!pos) return;

            const cardinality = simplex.length;
            const color = colors[cardinality % colors.length];

            // Create group for node and label
            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('data-simplex-idx', idx);
            g.style.cursor = 'pointer';

            // Circle
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', pos.x);
            circle.setAttribute('cy', pos.y);
            circle.setAttribute('r', nodeRadius);
            circle.setAttribute('fill', color);
            circle.setAttribute('stroke', '#333');
            circle.setAttribute('stroke-width', '1.5');
            circle.setAttribute('opacity', '0.85');

            // Label - convert neuron indices to readable format
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', pos.x);
            text.setAttribute('y', pos.y);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('dominant-baseline', 'middle');
            text.setAttribute('font-size', fontSize);
            text.setAttribute('font-weight', 'bold');
            text.setAttribute('fill', '#fff');
            text.setAttribute('pointer-events', 'none');

            // Generate label with proper neuron naming
            const neuronLabels = simplex.map(globalIdx => {
                const mapping = BinaryDisplay.getNeuronMapping(globalIdx);
                return mapping.label;
            });

            // Show full simplex if it fits, otherwise show count
            const simplexStr = neuronLabels.join(',');
            if (simplexStr.length <= 12) {
                text.textContent = simplexStr;
            } else if (simplex.length <= 3) {
                // For small sets, try to show abbreviated form
                text.textContent = neuronLabels.join(',');
            } else {
                text.textContent = `{${simplex.length}}`;
            }

            // Tooltip title with full information
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');

            // Build detailed neuron descriptions
            const neuronDescriptions = simplex.map(globalIdx => {
                const mapping = BinaryDisplay.getNeuronMapping(globalIdx);
                return `Hidden ${mapping.layer} Neuron ${mapping.neuron + 1}`;
            });

            const neuronText = simplex.length === 1 ? 'Neuron' : 'Neurons';
            title.textContent = `Cardinality ${cardinality} set: {${neuronLabels.join(', ')}}\n${neuronDescriptions.join(', ')}`;

            g.appendChild(circle);
            g.appendChild(text);
            g.appendChild(title);

            // Add hover effects
            g.addEventListener('mouseenter', () => {
                circle.setAttribute('opacity', '1');
                circle.setAttribute('stroke-width', '2.5');
            });

            g.addEventListener('mouseleave', () => {
                circle.setAttribute('opacity', '0.85');
                circle.setAttribute('stroke-width', '1.5');
            });

            // Add click handler for simplex selection
            g.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handleSimplexClick(simplex);
            });

            svg.appendChild(g);
        });

        // Clear container and add SVG
        container.innerHTML = '';
        container.appendChild(svg);
    },

    /**
     * Update the nerve complex visualization if it's currently active
     */
    updateIfActive() {
        if (this.isNerveView) {
            this.renderNerveComplex();
        }
    },

    /**
     * Handle click on a simplex node in the nerve complex
     * Selects the corresponding neurons in the decision boundary visualization
     * @param {Array<number>} simplex - Array of global neuron indices in this simplex
     */
    handleSimplexClick(simplex) {
        // Get the global neuron indices from the simplex
        const neuronIndices = simplex; // Array of global indices

        // Clear existing selections in DecisionBoundary
        if (typeof DecisionBoundary !== 'undefined') {
            DecisionBoundary.clearNeuronHighlights();
        }

        // Select the neurons in this simplex
        for (const globalIdx of neuronIndices) {
            if (typeof DecisionBoundary !== 'undefined') {
                DecisionBoundary.selectNeuron(globalIdx);
            }
        }

        // Update the decision boundary visualization
        if (typeof DecisionBoundary !== 'undefined') {
            DecisionBoundary.update();
        }

        // Update binary display to show highlighted neurons
        if (typeof BinaryDisplay !== 'undefined') {
            BinaryDisplay.update();
        }
    }
};

// Global function for onclick handler
function toggleDualNerve() {
    NerveComplexToggle.toggle();
}
