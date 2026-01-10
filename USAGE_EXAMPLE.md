# Betti Numbers - Usage Example

## Quick Start Guide

### Step 1: Set Up Neural Network
1. Open the application in your browser
2. Configure a simple network:
   - Input Size: 2
   - Hidden Layers: 4,3
   - Output Size: 1
3. Click "Rebuild Network"

### Step 2: Generate Training Data
1. Enter a target function, e.g., `x*x + y*y - 1.5`
2. Set Sample Points: 50
3. Click "Generate Training Data"

### Step 3: Train the Network (Optional)
1. Click "Train 1 Step" or "Train 100 Steps"
2. Watch the network learn the function
3. The polyhedral decomposition will update

### Step 4: View Nerve Complex
1. Locate the "Dual Graph / Nerve Complex" window (right side, below the polyhedral decomposition)
2. Click the toggle switch to switch from "Dual Graph" to "Nerve Complex"
3. The visualization will show the poset of neuron activation regions

### Step 5: Compute Betti Numbers
1. Below the nerve complex visualization, you'll see a new button: "Compute Betti Numbers"
2. Click the button
3. Wait a moment for the computation (you'll see "Computing...")
4. View the results!

## Understanding the Results

### Example Output
```
Betti Numbers of Order Complex:
β₀ = 1 (connected components)
β₁ = 3 (1-dimensional holes)
β₂ = 0 (2-dimensional holes)

Poset: 15 elements, 12 edges
Order complex: up to dimension 2
```

### Interpretation

**β₀ = 1**: The activation structure is connected
- There is one connected component
- All regions are reachable from each other through containment relations

**β₁ = 3**: There are three 1-dimensional holes (loops)
- These represent cycles in the containment structure
- For example: Region A ⊂ B, B ⊂ C, C ⊂ D, but A and D are not directly related
- Indicates non-trivial topological structure in how neurons activate

**β₂ = 0**: No 2-dimensional voids
- The structure doesn't have any "hollow" 2D regions
- This is common for smaller networks

## Different Network Configurations

### Simple Network (2-2-1)
Expected: Small poset, likely β₀ = 1, β₁ = 0 or 1

### Complex Network (2-8-8-1)
Expected: Larger poset, potentially higher Betti numbers
- More neurons means more possible activation patterns
- May see β₁ ≥ 2 or even β₂ > 0

### XOR Problem
Classic test case:
- Network: 2-2-1
- Function: `(x > 0) !== (y > 0)` (XOR logic)
- Expected: Non-trivial topology with β₁ > 0

## What Do Betti Numbers Tell Us?

### β₀ (Connected Components)
- **= 1**: Network's activation regions form one connected structure
- **> 1**: Disconnected activation patterns (unusual, may indicate issues)

### β₁ (Loops)
- **= 0**: Tree-like structure, no cycles
- **> 0**: Network has cyclic activation patterns
  - Can indicate redundancy in neuron usage
  - Or complex learned feature interactions

### β₂ and Higher (Voids)
- **= 0**: No higher-dimensional holes (most common)
- **> 0**: Very complex topological structure
  - Rare in practice
  - May indicate highly non-linear feature space

## Practical Applications

### 1. Network Analysis
- Compare Betti numbers before and after training
- Higher β₁ after training may indicate learned complexity

### 2. Architecture Comparison
- Compare different network architectures on same problem
- Networks with similar performance may have different topologies

### 3. Debugging
- Unusually high or low Betti numbers may indicate:
  - Dead neurons (some regions never activate)
  - Redundant neurons (creating unnecessary complexity)
  - Interesting learned structure

### 4. Research
- Study how topology evolves during training
- Relate topological features to generalization performance
- Explore connections to computational topology

## Performance Notes

### Computation Time
- Small networks (< 20 simplices): < 100ms
- Medium networks (20-50 simplices): 100-500ms
- Large networks (> 50 simplices): 500ms - 2s

### When to Use
- After training to analyze final structure
- At key points during training to track evolution
- When comparing different architectures

### Caching
Results are cached - clicking again won't recompute unless:
- Network architecture changes
- Network is retrained
- Forward pass generates new activations

## Troubleshooting

### "No simplices in poset"
- Perform a forward pass first
- Network may need to be trained
- Try generating training data

### "No neural network available"
- Rebuild the network
- Check that network is properly initialized

### Very high Betti numbers
- May indicate sampling captured many activation patterns
- Try training more to converge to stable structure
- Consider if this reflects true complexity or noise

### All zeros except β₀ = 1
- Common for very simple networks or untrained networks
- Tree-like structure with no loops
- Consider training more or using more complex architecture

## Advanced Usage

### Comparing Topologies
1. Save network state
2. Compute Betti numbers
3. Train further
4. Compute again
5. Compare evolution

### Export and Analysis
- Copy results for external analysis
- Track Betti numbers over training epochs
- Correlate with loss/accuracy metrics

### Integration with Research
- Use as features for meta-learning
- Study topology-performance relationships
- Explore persistent homology extensions
