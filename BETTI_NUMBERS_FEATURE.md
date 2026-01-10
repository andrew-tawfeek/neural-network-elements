# Betti Numbers Feature Implementation

## Overview
This feature adds the ability to compute the Betti numbers of the order complex of the poset visualization in the nerve complex view.

## Files Modified/Created

### New Files
- `js/betti-numbers.js` - Complete implementation of Betti number computation

### Modified Files
- `index.html` - Added UI components (button and display area)
- `js/nerve-complex-toggle.js` - Updated to show/hide Betti numbers section

## Mathematical Background

### Order Complex
Given a poset P (partially ordered set), the order complex Δ(P) is a simplicial complex where:
- **Vertices**: Elements of the poset
- **k-simplices**: Chains (totally ordered sets) of k+1 elements: p₀ < p₁ < ... < pₖ

In our case, the poset consists of:
- Elements: Sets of neurons that can be simultaneously active (simplices from nerve complex)
- Order relation: Containment (⊆)

### Betti Numbers
Betti numbers are topological invariants that count "holes" in different dimensions:
- **β₀**: Number of connected components
- **β₁**: Number of 1-dimensional holes (loops)
- **β₂**: Number of 2-dimensional holes (voids)
- **βₖ**: Rank of the k-th homology group Hₖ

## Algorithm Implementation

### 1. Build Order Complex (`buildOrderComplex`)
- Input: Poset with simplices and containment edges (Hasse diagram)
- Process: Find all chains using DFS
  - A chain is a path in the Hasse diagram
  - Each chain represents a totally ordered subset
- Output: Chains organized by dimension

### 2. Compute Boundary Matrices (`computeBoundaryMatrix`)
- For dimension k, creates boundary operator ∂ₖ: Cₖ → Cₖ₋₁
- Maps k-chains to (k-1)-chains
- Boundary of chain [s₀, s₁, ..., sₖ] is Σᵢ (-1)ⁱ [s₀, ..., ŝᵢ, ..., sₖ]
- Works over Z/2Z (mod 2 arithmetic) for simplicity, so (-1)ⁱ = 1

### 3. Compute Matrix Rank (`computeMatrixRank`)
- Uses Gaussian elimination over Z/2Z
- Row reduction to echelon form
- Counts pivot rows to determine rank

### 4. Calculate Betti Numbers (`computeHomology`)
- Formula: **βₖ = dim(Cₖ) - rank(∂ₖ) - rank(∂ₖ₊₁)**
- This equals: rank(ker(∂ₖ)) - rank(im(∂ₖ₊₁))
- Which is the rank of the k-th homology group: Hₖ = ker(∂ₖ) / im(∂ₖ₊₁)

## User Interface

### Location
The feature appears in the "Dual Graph / Nerve Complex" window, underneath the nerve complex visualization.

### Visibility
- **Hidden** when "Dual Graph" view is active
- **Visible** when "Nerve Complex" view is active

### Components
1. **Button**: "Compute Betti Numbers"
   - Blue background (#3498db)
   - Hover effect (darker blue #2980b9)
   - Displays "Computing..." during calculation

2. **Display Area**:
   - Shows computed Betti numbers with interpretations
   - Format: β₀ = 1 (connected components), β₁ = 3 (1-dimensional holes), etc.
   - Includes summary statistics (poset size, number of edges)
   - Uses subscript formatting for mathematical notation

## Usage

1. Build or load a neural network
2. Perform a forward pass to generate activations
3. Switch to "Nerve Complex" view using the toggle switch
4. Click "Compute Betti Numbers" button
5. View results in the display area below

## Edge Cases Handled

- Empty poset: Shows appropriate error message
- No hidden layers: Returns error
- Single element poset: Returns β₀ = 1, all others = 0
- Large posets: Uses setTimeout to prevent UI freezing
- Repeated clicks: Prevents multiple simultaneous computations

## Performance Considerations

- Caching: Results are cached to avoid recomputation
- Asynchronous: Uses setTimeout for UI responsiveness
- Z/2Z arithmetic: Simplifies computation (no need to track signs)
- Efficient rank computation: Gaussian elimination is O(n³)

## Testing Recommendations

### Simple Test Cases
1. **Single neuron**: Should have β₀ = 1, all others = 0
2. **Two neurons with overlap**: Should show connected structure
3. **Three neurons in triangle**: May show β₁ ≥ 1 if they form a loop
4. **XOR problem**: Classic neural network with interesting topology

### Expected Outputs
- β₀ should typically be 1 (one connected component)
- β₁ > 0 indicates presence of loops/cycles in activation structure
- Higher Betti numbers indicate more complex topological features

## Code Quality

- **Documented**: All functions have JSDoc comments
- **Modular**: Separated concerns (computation, display, UI)
- **Error handling**: Try-catch blocks and validation
- **Standards compliant**: Modern ES6+ JavaScript
- **Accessible**: Clear error messages and loading states

## Integration Points

- **BinaryDisplay.computeUniversalNerveComplex()**: Source of poset data
- **NerveComplexToggle**: Controls visibility of Betti numbers section
- **MathJax**: Optional rendering of mathematical notation (if available)

## Future Enhancements

Potential improvements:
1. Visualization of representative cycles for β₁
2. Computation over Z (integers) instead of Z/2Z for torsion information
3. Persistent homology for multi-scale analysis
4. Export Betti numbers with nerve complex data
5. Progress bar for large computations
6. Detailed breakdown by dimension with collapsible sections
