// Neural Network Class and Core Functionality
class MultiLayerNetwork {
    constructor(a) {
        this.architecture = a
        this.layers = a.length
        this.weights = []
        this.biases = []
        this.activations = []
        this.zValues = []

        // Adam optimizer state
        this.adamM_w = [] // First moment (momentum) for weights
        this.adamV_w = [] // Second moment (RMSprop) for weights
        this.adamM_b = [] // First moment for biases
        this.adamV_b = [] // Second moment for biases
        this.adamT = 0    // Time step for bias correction

        // Optimizer hyperparameters
        this.beta1 = 0.9      // Momentum decay
        this.beta2 = 0.999    // RMSprop decay
        this.epsilon = 1e-8   // Numerical stability
        this.gradClipThreshold = 5.0  // Gradient clipping threshold

        // Initialize weights with Xavier/He initialization
        for (let i = 0; i < this.layers - 1; i++) {
            this.weights.push(this.heInitialization(a[i + 1], a[i]))
            this.biases.push(this.zeros(a[i + 1], 1))

            // Initialize Adam moments
            this.adamM_w.push(this.zeros(a[i + 1], a[i]))
            this.adamV_w.push(this.zeros(a[i + 1], a[i]))
            this.adamM_b.push(this.zeros(a[i + 1], 1))
            this.adamV_b.push(this.zeros(a[i + 1], 1))
        }
    }

    zeros(r, c) {
        const m = []
        for (let i = 0; i < r; i++) {
            m[i] = []
            for (let j = 0; j < c; j++) m[i][j] = 0
        }
        return m
    }

    heInitialization(r, c) {
        // He initialization for ReLU networks: std = sqrt(2/n_in)
        const std = Math.sqrt(2.0 / c)
        const m = []
        for (let i = 0; i < r; i++) {
            m[i] = []
            for (let j = 0; j < c; j++) {
                // Box-Muller transform for normal distribution
                const u1 = Math.random()
                const u2 = Math.random()
                m[i][j] = std * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
            }
        }
        return m
    }
    
    rand(r, c) {
        const m = []
        for (let i = 0; i < r; i++) {
            m[i] = []
            for (let j = 0; j < c; j++)m[i][j] = (Math.random() - .5) * 2
        }
        return m
    }
    
    relu(x) { return Math.max(0, x) }
    reluDer(x) { return x > 0 ? 1 : 0 }
    
    mul(A, B) {
        const r = []
        for (let i = 0; i < A.length; i++) {
            r[i] = []
            for (let j = 0; j < B[0].length; j++) {
                let s = 0
                for (let k = 0; k < B.length; k++)s += A[i][k] * B[k][j]
                r[i][j] = s
            }
        }
        return r
    }
    
    add(A, B) {
        const r = []
        for (let i = 0; i < A.length; i++) {
            r[i] = []
            for (let j = 0; j < A[0].length; j++)r[i][j] = A[i][j] + B[i][j]
        }
        return r
    }
    
    forward(input) {
        this.activations = []
        this.zValues = []
        this.activations.push(input.map(x => [x]))
        for (let l = 0; l < this.layers - 1; l++) {
            const z = this.add(this.mul(this.weights[l], this.activations[l]), this.biases[l])
            this.zValues.push(z)
            let a
            if (l === this.layers - 2) {
                a = z
            } else {
                a = z.map(row => row.map(x => this.relu(x)))
            }
            this.activations.push(a)
        }
        return this.activations[this.activations.length - 1].map(row => row[0])
    }
    
    backward(input, target, lr) {
        const out = this.forward(input);
        const err = [];
        let loss = 0;
        for (let i = 0; i < out.length; i++) {
            err[i] = out[i] - target[i];
            loss += err[i] * err[i];
        }
        loss /= 2;

        // Compute gradients using backpropagation
        const gradW = []
        const gradB = []

        // Output layer delta (linear activation)
        let delta = err.map(e => [e]);

        for (let l = this.layers - 2; l >= 0; l--) {
            // Compute gradients for this layer
            const dW = [];
            const db = [];
            for (let i = 0; i < this.architecture[l + 1]; i++) {
                dW[i] = [];
                db[i] = [delta[i][0]];
                for (let j = 0; j < this.architecture[l]; j++) {
                    dW[i][j] = delta[i][0] * this.activations[l][j][0];
                }
            }

            gradW.unshift(dW)
            gradB.unshift(db)

            // Compute delta for previous layer (if not input)
            if (l > 0) {
                const newDelta = [];
                for (let j = 0; j < this.architecture[l]; j++) {
                    let sum = 0;
                    for (let i = 0; i < this.architecture[l + 1]; i++) {
                        sum += delta[i][0] * this.weights[l][i][j];
                    }
                    // Apply ReLU derivative to zValues
                    sum *= this.reluDer(this.zValues[l - 1][j][0]);
                    newDelta[j] = [sum];
                }
                delta = newDelta;
            }
        }

        // Apply gradient clipping
        this.clipGradients(gradW, gradB)

        // Apply Adam optimizer update
        this.adamUpdate(gradW, gradB, lr)

        return loss;
    }

    clipGradients(gradW, gradB) {
        // Global norm clipping
        let totalNorm = 0

        // Calculate total norm
        for (let l = 0; l < gradW.length; l++) {
            for (let i = 0; i < gradW[l].length; i++) {
                for (let j = 0; j < gradW[l][i].length; j++) {
                    totalNorm += gradW[l][i][j] * gradW[l][i][j]
                }
            }
            for (let i = 0; i < gradB[l].length; i++) {
                totalNorm += gradB[l][i][0] * gradB[l][i][0]
            }
        }
        totalNorm = Math.sqrt(totalNorm)

        // Clip if necessary
        if (totalNorm > this.gradClipThreshold) {
            const clipCoef = this.gradClipThreshold / (totalNorm + 1e-6)
            for (let l = 0; l < gradW.length; l++) {
                for (let i = 0; i < gradW[l].length; i++) {
                    for (let j = 0; j < gradW[l][i].length; j++) {
                        gradW[l][i][j] *= clipCoef
                    }
                }
                for (let i = 0; i < gradB[l].length; i++) {
                    gradB[l][i][0] *= clipCoef
                }
            }
        }
    }

    adamUpdate(gradW, gradB, lr) {
        // Increment time step
        this.adamT++

        // Bias correction coefficients
        const biasCorr1 = 1 - Math.pow(this.beta1, this.adamT)
        const biasCorr2 = 1 - Math.pow(this.beta2, this.adamT)

        // Update parameters layer by layer
        for (let l = 0; l < this.layers - 1; l++) {
            // Update weights
            for (let i = 0; i < this.architecture[l + 1]; i++) {
                for (let j = 0; j < this.architecture[l]; j++) {
                    const g = gradW[l][i][j]

                    // Update moments
                    this.adamM_w[l][i][j] = this.beta1 * this.adamM_w[l][i][j] + (1 - this.beta1) * g
                    this.adamV_w[l][i][j] = this.beta2 * this.adamV_w[l][i][j] + (1 - this.beta2) * g * g

                    // Bias-corrected moments
                    const mHat = this.adamM_w[l][i][j] / biasCorr1
                    const vHat = this.adamV_w[l][i][j] / biasCorr2

                    // Update weight
                    this.weights[l][i][j] -= lr * mHat / (Math.sqrt(vHat) + this.epsilon)
                }
            }

            // Update biases
            for (let i = 0; i < this.architecture[l + 1]; i++) {
                const g = gradB[l][i][0]

                // Update moments
                this.adamM_b[l][i][0] = this.beta1 * this.adamM_b[l][i][0] + (1 - this.beta1) * g
                this.adamV_b[l][i][0] = this.beta2 * this.adamV_b[l][i][0] + (1 - this.beta2) * g * g

                // Bias-corrected moments
                const mHat = this.adamM_b[l][i][0] / biasCorr1
                const vHat = this.adamV_b[l][i][0] / biasCorr2

                // Update bias
                this.biases[l][i][0] -= lr * mHat / (Math.sqrt(vHat) + this.epsilon)
            }
        }
    }
    
    /*
    - Only return hidden layers' binary vectors
    - For full vector: concatenate only hidden layers (i=1 to len-2), skip input (0) and output (last)
    */
    getBinaryState() {
        if (!this.activations || this.activations.length === 0) return []
        let fullVector = []
        for (let l = 1; l < this.activations.length - 1; l++) { // Skip input (0) and output (last)
            const binaryVec = this.activations[l].map(neuron => neuron[0] > 0 ? 1 : 0)
            fullVector = fullVector.concat(binaryVec)
        }
        return fullVector
    }

    saveState() {
        // Deep copy the current state of the network including optimizer state
        return {
            architecture: [...this.architecture],
            weights: this.weights.map(layer =>
                layer.map(neuron => [...neuron])
            ),
            biases: this.biases.map(layer =>
                layer.map(bias => [...bias])
            ),
            activations: this.activations ? this.activations.map(layer =>
                layer.map(activation => [...activation])
            ) : [],
            zValues: this.zValues ? this.zValues.map(layer =>
                layer.map(z => [...z])
            ) : [],
            // Save Adam optimizer state
            adamM_w: this.adamM_w.map(layer => layer.map(row => [...row])),
            adamV_w: this.adamV_w.map(layer => layer.map(row => [...row])),
            adamM_b: this.adamM_b.map(layer => layer.map(row => [...row])),
            adamV_b: this.adamV_b.map(layer => layer.map(row => [...row])),
            adamT: this.adamT
        };
    }

    loadState(state) {
        // Restore the network to a previous state including optimizer state
        this.architecture = [...state.architecture];
        this.layers = this.architecture.length;
        this.weights = state.weights.map(layer =>
            layer.map(neuron => [...neuron])
        );
        this.biases = state.biases.map(layer =>
            layer.map(bias => [...bias])
        );
        this.activations = state.activations ? state.activations.map(layer =>
            layer.map(activation => [...activation])
        ) : [];
        this.zValues = state.zValues ? state.zValues.map(layer =>
            layer.map(z => [...z])
        ) : [];

        // Restore Adam optimizer state if available
        if (state.adamM_w) {
            this.adamM_w = state.adamM_w.map(layer => layer.map(row => [...row]));
            this.adamV_w = state.adamV_w.map(layer => layer.map(row => [...row]));
            this.adamM_b = state.adamM_b.map(layer => layer.map(row => [...row]));
            this.adamV_b = state.adamV_b.map(layer => layer.map(row => [...row]));
            this.adamT = state.adamT || 0;
        } else {
            // Initialize Adam state if not present in saved state (backward compatibility)
            this.adamM_w = [];
            this.adamV_w = [];
            this.adamM_b = [];
            this.adamV_b = [];
            this.adamT = 0;
            for (let i = 0; i < this.layers - 1; i++) {
                this.adamM_w.push(this.zeros(this.architecture[i + 1], this.architecture[i]));
                this.adamV_w.push(this.zeros(this.architecture[i + 1], this.architecture[i]));
                this.adamM_b.push(this.zeros(this.architecture[i + 1], 1));
                this.adamV_b.push(this.zeros(this.architecture[i + 1], 1));
            }
        }
    }
}

// Network management functions
function createNetwork() {
    const inputSize = parseInt(document.getElementById('input-size').value)
    const hiddenLayersStr = document.getElementById('hidden-layers').value
    const outputSize = parseInt(document.getElementById('output-size').value)
    const hiddenLayers = hiddenLayersStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0)
    const architecture = [inputSize, ...hiddenLayers, outputSize]
    window.net = new MultiLayerNetwork(architecture)
    createInputs()
    createWeights()
    forward() // Call forward pass to initialize activations
    NetworkVisualization.draw()
    BinaryDisplay.update()
    DecisionBoundary.update()
    PlotManager.initialize()
    window.data = []
    
    // Reset loss graph when creating new network
    if (typeof LossGraph !== 'undefined') {
        LossGraph.clearHistory();
    }
    
    // Reset training step counter
    if (typeof TrainingManager !== 'undefined') {
        TrainingManager.stepCounter = 0;
    }
    
    const targetFunctionControls = document.getElementById('target-function-controls');
    if (targetFunctionControls) {
        targetFunctionControls.style.display = window.net.architecture[0] === 2 ? 'block' : 'none';
    }
}

function createInputs() {
    const c = document.getElementById('input-controls')
    c.innerHTML = ''
    for (let i = 0; i < window.net.architecture[0]; i++) {
        const d = document.createElement('div')
        d.className = 'row'
        d.innerHTML = `<label>Input ${i + 1}:</label><div class="info" data-tip="Input value for neuron ${i + 1}">i</div><input type="number" id="input-${i}" value="0" step="0.1" onchange="forward()">`
        c.appendChild(d)
    }
}

function createWeights() {
    const c = document.getElementById('weight-controls')
    c.innerHTML = ''

    for (let l = 0; l < window.net.layers - 1; l++) {
        const layerDiv = document.createElement('div')
        layerDiv.className = 'layer'
        const layerName = l === window.net.layers - 2 ? 'Output' : `Hidden ${l + 1}`

        // Layer title
        layerDiv.innerHTML = `<div class="layer-title">${layerName} Layer ($W^{${l + 1}}$, $b^{${l + 1}}$)</div>`

        // Weights section
        const weightsDiv = document.createElement('div')
        weightsDiv.innerHTML = `<div style="font-weight:500;color:#2c3e50;margin:10px 0">Weights:</div>`

        for (let i = 0; i < window.net.architecture[l + 1]; i++) {
            for (let j = 0; j < window.net.architecture[l]; j++) {
                const r = document.createElement('div')
                r.className = 'weight-row'
                const v = window.net.weights[l][i][j]
                r.innerHTML = `
            <span class="weight-label" id="label-w${l}-${i}-${j}">$W^{${l + 1}}_{${i + 1},${j + 1}}$:</span>
            <div class="info" data-tip="Weight from layer ${l} neuron ${j + 1} to layer ${l + 1} neuron ${i + 1}">i</div>
            <div class="slider-container">
                <input type="range" min="-3" max="3" step="0.1" value="${v.toFixed(1)}" 
                       oninput="updateW(${l},${i},${j},this.value)" id="s-w${l}-${i}-${j}">
                <input type="number" step="0.1" value="${v.toFixed(2)}" 
                       onchange="updateWN(${l},${i},${j},this.value)" id="n-w${l}-${i}-${j}">
            </div>`
                weightsDiv.appendChild(r)
            }
        }
        layerDiv.appendChild(weightsDiv)

        // Biases section
        const biasesDiv = document.createElement('div')
        biasesDiv.innerHTML = `<div style="font-weight:500;color:#2c3e50;margin:10px 0">Biases:</div>`

        for (let i = 0; i < window.net.architecture[l + 1]; i++) {
            const b = document.createElement('div')
            b.className = 'weight-row'
            const bv = window.net.biases[l][i][0]
            b.innerHTML = `
        <span class="weight-label" id="label-b${l}-${i}">$b^{${l + 1}}_{${i + 1}}$:</span>
        <div class="info" data-tip="Bias for layer ${l + 1} neuron ${i + 1}">i</div>
        <div class="slider-container">
            <input type="range" min="-3" max="3" step="0.1" value="${bv.toFixed(1)}" 
                   oninput="updateB(${l},${i},this.value)" id="s-b${l}-${i}">
            <input type="number" step="0.1" value="${bv.toFixed(2)}" 
                   onchange="updateBN(${l},${i},this.value)" id="n-b${l}-${i}">
        </div>`
            biasesDiv.appendChild(b)
        }
        layerDiv.appendChild(biasesDiv)
        c.appendChild(layerDiv)
    }

    // Trigger MathJax to rerender the math notation
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
}

function updateW(l, i, j, v) {
    const val = parseFloat(v)
    window.net.weights[l][i][j] = val
    document.getElementById(`n-w${l}-${i}-${j}`).value = val.toFixed(2)
    forward()
}

function updateWN(l, i, j, v) {
    const val = parseFloat(v)
    window.net.weights[l][i][j] = val
    document.getElementById(`s-w${l}-${i}-${j}`).value = val.toFixed(1)
    forward()
}

function updateB(l, i, v) {
    const val = parseFloat(v)
    window.net.biases[l][i][0] = val
    document.getElementById(`n-b${l}-${i}`).value = val.toFixed(2)
    forward()
}

function updateBN(l, i, v) {
    const val = parseFloat(v)
    window.net.biases[l][i][0] = val
    document.getElementById(`s-b${l}-${i}`).value = val.toFixed(1)
    forward()
}

function forward() {
    const inputs = []
    for (let i = 0; i < window.net.architecture[0]; i++) {
        inputs.push(parseFloat(document.getElementById(`input-${i}`).value) || 0)
    }
    const out = window.net.forward(inputs)
    document.getElementById('output').innerHTML = `<strong>Input:</strong> [${inputs.map(x => x.toFixed(2)).join(', ')}]<br><strong>Output:</strong> [${out.map(x => x.toFixed(3)).join(', ')}]`
    NetworkVisualization.draw(false) // Regular update with new network state
    BinaryDisplay.update()
}

function randomize() {
    for (let l = 0; l < window.net.layers - 1; l++) {
        window.net.weights[l] = window.net.rand(window.net.architecture[l + 1], window.net.architecture[l])
        window.net.biases[l] = window.net.rand(window.net.architecture[l + 1], 1)
    }
    createWeights()
    forward()
}

function zero() {
    for (let l = 0; l < window.net.layers - 1; l++) {
        // Set all weights to zero
        for (let i = 0; i < window.net.architecture[l + 1]; i++) {
            for (let j = 0; j < window.net.architecture[l]; j++) {
                window.net.weights[l][i][j] = 0
            }
        }
        // Set all biases to zero
        for (let i = 0; i < window.net.architecture[l + 1]; i++) {
            window.net.biases[l][i][0] = 0
        }
    }
    createWeights()
    forward()
}

function reset() {
    createNetwork()
}
