/**
 * Binary Classification Dataset Generator Module
 *
 * Provides classic binary classification patterns for testing neural networks:
 * - Annuli (Concentric Circles): Two concentric rings with different classes
 * - Two Clusters: Two separated Gaussian clusters
 * - XOR Pattern: Four quadrants alternating between classes
 * - Spiral: Two intertwined spirals
 * - Checkerboard: Grid pattern with alternating classes
 * - Half Plane: Points above/below a dividing line
 *
 * All generators create balanced datasets with equal numbers of each class.
 * Data format: { inputs: [x, y], targets: [class] }
 * where class is 0 or 1 for binary classification.
 */

const ClassificationDatasets = {
    /**
     * Main function to generate classification data based on user selection
     * Reads values from UI controls and updates the training data
     */
    generateClassificationData() {
        const datasetType = document.getElementById('classification-dataset').value;
        const pointsPerClass = parseInt(document.getElementById('classification-points').value) || 50;

        // Validate that we have a 2D input network
        if (!window.net || window.net.architecture[0] !== 2) {
            document.getElementById('error').innerHTML =
                'Classification datasets require a 2D input network. Set Input Size to 2 and rebuild the network.';
            return;
        }

        let dataPoints = [];
        let datasetName = '';

        try {
            switch(datasetType) {
                case 'annuli':
                    dataPoints = this.generateAnnuli(pointsPerClass);
                    datasetName = 'Annuli (Concentric Circles)';
                    break;
                case 'clusters':
                    dataPoints = this.generateTwoClusters(pointsPerClass);
                    datasetName = 'Two Clusters';
                    break;
                case 'xor':
                    dataPoints = this.generateXOR(pointsPerClass);
                    datasetName = 'XOR Pattern';
                    break;
                case 'spiral':
                    dataPoints = this.generateSpiral(pointsPerClass);
                    datasetName = 'Spiral';
                    break;
                case 'checkerboard':
                    dataPoints = this.generateCheckerboard(pointsPerClass);
                    datasetName = 'Checkerboard';
                    break;
                case 'halfplane':
                    dataPoints = this.generateHalfPlane(pointsPerClass);
                    datasetName = 'Half Plane';
                    break;
                default:
                    throw new Error('Unknown dataset type: ' + datasetType);
            }

            // Shuffle the data points for better training
            dataPoints = this.shuffleArray(dataPoints);

            // Replace current training data with generated data
            window.data = dataPoints;

            // Reset training step counter
            if (TrainingManager) {
                TrainingManager.stepCounter = 0;
            }

            // Clear loss graph
            if (typeof LossGraph !== 'undefined') {
                LossGraph.clearHistory();
            }

            // Update the data display
            updateDataDisplay();

            // Update success message
            const totalPoints = dataPoints.length;
            const class0Count = dataPoints.filter(p => p.targets[0] === 0).length;
            const class1Count = dataPoints.filter(p => p.targets[0] === 1).length;

            document.getElementById('error').innerHTML =
                `Generated ${datasetName}: ${totalPoints} points (Class 0: ${class0Count}, Class 1: ${class1Count})`;

            // Update the plot if plot functionality exists
            if (typeof updatePlot === 'function') {
                updatePlot();
            }

            // Set a reasonable learning rate for classification
            document.getElementById('learning-rate').value = '0.01';

        } catch (error) {
            document.getElementById('error').innerHTML =
                'Error generating dataset: ' + error.message;
            console.error('Classification dataset generation error:', error);
        }
    },

    /**
     * Generate Annuli (Concentric Circles) dataset
     * Inner circle is class 0, outer ring is class 1
     * This tests the network's ability to learn non-linearly separable patterns
     */
    generateAnnuli(pointsPerClass) {
        const data = [];

        // Inner circle (class 0) - radius between 0.5 and 1
        for (let i = 0; i < pointsPerClass; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const r = 0.5 + Math.random() * 0.5;
            const x = r * Math.cos(angle);
            const y = r * Math.sin(angle);
            data.push({ inputs: [x, y], targets: [0] });
        }

        // Outer ring (class 1) - radius between 1.5 and 2
        for (let i = 0; i < pointsPerClass; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const r = 1.5 + Math.random() * 0.5;
            const x = r * Math.cos(angle);
            const y = r * Math.sin(angle);
            data.push({ inputs: [x, y], targets: [1] });
        }

        return data;
    },

    /**
     * Generate Two Clusters dataset
     * Two separated Gaussian clusters at opposite corners
     * This is linearly separable and good for testing basic classification
     */
    generateTwoClusters(pointsPerClass) {
        const data = [];

        // Cluster 1 (class 0) - centered at (-1, -1)
        for (let i = 0; i < pointsPerClass; i++) {
            const x = -1 + (Math.random() - 0.5) * 1.0;
            const y = -1 + (Math.random() - 0.5) * 1.0;
            data.push({ inputs: [x, y], targets: [0] });
        }

        // Cluster 2 (class 1) - centered at (1, 1)
        for (let i = 0; i < pointsPerClass; i++) {
            const x = 1 + (Math.random() - 0.5) * 1.0;
            const y = 1 + (Math.random() - 0.5) * 1.0;
            data.push({ inputs: [x, y], targets: [1] });
        }

        return data;
    },

    /**
     * Generate XOR Pattern dataset
     * Four quadrants with alternating classes - classic XOR problem
     * This is a standard test for non-linear classification
     * Requires hidden layers to solve
     */
    generateXOR(pointsPerClass) {
        const data = [];
        const perQuadrant = Math.floor(pointsPerClass / 2);

        // Class 0: top-left and bottom-right quadrants
        for (let i = 0; i < perQuadrant; i++) {
            // Top-left quadrant
            data.push({
                inputs: [-2 + Math.random() * 2, Math.random() * 2],
                targets: [0]
            });
            // Bottom-right quadrant
            data.push({
                inputs: [Math.random() * 2, -2 + Math.random() * 2],
                targets: [0]
            });
        }

        // Class 1: top-right and bottom-left quadrants
        for (let i = 0; i < perQuadrant; i++) {
            // Top-right quadrant
            data.push({
                inputs: [Math.random() * 2, Math.random() * 2],
                targets: [1]
            });
            // Bottom-left quadrant
            data.push({
                inputs: [-2 + Math.random() * 2, -2 + Math.random() * 2],
                targets: [1]
            });
        }

        return data;
    },

    /**
     * Generate Spiral dataset
     * Two intertwined spirals with different classes
     * This is one of the most challenging binary classification problems
     * Requires significant network capacity to separate
     */
    generateSpiral(pointsPerClass) {
        const data = [];

        for (let i = 0; i < pointsPerClass; i++) {
            // Parameter t controls spiral progression (2 full rotations)
            const t = i / pointsPerClass * 4 * Math.PI;
            const r = t / (4 * Math.PI) * 2; // radius grows from 0 to 2

            // Add small random noise for realism
            const noise = () => (Math.random() - 0.5) * 0.2;

            // Spiral 1 (class 0)
            const x0 = r * Math.cos(t) + noise();
            const y0 = r * Math.sin(t) + noise();
            data.push({ inputs: [x0, y0], targets: [0] });

            // Spiral 2 (class 1) - offset by π radians
            const x1 = r * Math.cos(t + Math.PI) + noise();
            const y1 = r * Math.sin(t + Math.PI) + noise();
            data.push({ inputs: [x1, y1], targets: [1] });
        }

        return data;
    },

    /**
     * Generate Checkerboard dataset
     * Grid pattern with alternating classes like a checkerboard
     * Tests the network's ability to learn complex periodic boundaries
     */
    generateCheckerboard(pointsPerClass) {
        const data = [];
        const totalPoints = pointsPerClass * 2;

        for (let i = 0; i < totalPoints; i++) {
            // Generate random point in range -2 to 2
            const x = (Math.random() - 0.5) * 4;
            const y = (Math.random() - 0.5) * 4;

            // Determine class based on checkerboard pattern
            // Shift coordinates to make them non-negative, then floor to get cell indices
            const xCell = Math.floor(x + 2);
            const yCell = Math.floor(y + 2);

            // XOR of cell indices determines the checkerboard pattern
            const target = (xCell + yCell) % 2;

            data.push({ inputs: [x, y], targets: [target] });
        }

        return data;
    },

    /**
     * Generate Half Plane dataset
     * Points are divided by a diagonal line (y = x)
     * This is the simplest linearly separable pattern
     * A single perceptron can solve this
     */
    generateHalfPlane(pointsPerClass) {
        const data = [];

        for (let i = 0; i < pointsPerClass * 2; i++) {
            // Generate random point in range -2 to 2
            const x = (Math.random() - 0.5) * 4;
            const y = (Math.random() - 0.5) * 4;

            // Divide by diagonal line y = x
            // Points above the line are class 1, below are class 0
            const target = y > x ? 1 : 0;

            data.push({ inputs: [x, y], targets: [target] });
        }

        return data;
    },

    /**
     * Utility function to shuffle an array using Fisher-Yates algorithm
     * This ensures training data is in random order for better learning
     */
    shuffleArray(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }
};

/**
 * Global function for backward compatibility and HTML onclick handler
 * Called when user clicks "Generate Classification Data" button
 */
function generateClassificationData() {
    ClassificationDatasets.generateClassificationData();
}
