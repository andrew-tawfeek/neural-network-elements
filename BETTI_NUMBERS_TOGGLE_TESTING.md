# Betti Numbers Toggle - Testing Guide

## Overview
The Betti Numbers feature has been converted from a manual "Compute" button to an automatic toggle switch that updates whenever the poset changes.

## Files Modified

1. **index.html** (lines 229-240)
   - Replaced button with toggle switch
   - Added checkbox with id `betti-numbers-toggle`

2. **js/betti-numbers.js**
   - Added state tracking: `showBettiNumbers`, `lastPosetState`
   - Added `hasPosetChanged()` method for change detection
   - Added `updateIfNeeded()` method for automatic updates
   - Added `computeAndDisplay()` method that accepts optional poset parameter
   - Added `toggleBettiNumbers()` global function for UI integration

3. **js/nerve-complex-toggle.js** (lines 295-299)
   - Added automatic Betti numbers update after rendering nerve complex
   - Calls `BettiNumbers.updateIfNeeded()` with the current poset

## Testing Instructions

### 1. Basic Toggle Functionality

**Test: Toggle On**
1. Open the application in a browser
2. Build a network with 2 inputs (e.g., default: 2 inputs, 4,3 hidden, 1 output)
3. Click "Forward Pass" to compute initial state
4. Click the toggle to switch from "Dual Graph" to "Nerve Complex"
5. The Betti Numbers section should appear below the nerve complex
6. Toggle the "Show Betti Numbers" switch to ON
7. Expected: Betti numbers should compute and display automatically
8. Display should show "Computing..." briefly, then results

**Test: Toggle Off**
1. With Betti numbers displayed, toggle the switch to OFF
2. Expected: Display area clears immediately
3. No computation should occur

### 2. Auto-Update on Poset Changes

**Test: Network Changes Trigger Update**
1. Toggle "Show Betti Numbers" to ON
2. Change network input values and click "Forward Pass"
3. Expected: If the binary state changes, Betti numbers should auto-update
4. Watch for "Computing..." message during update

**Test: Training Updates**
1. Generate some training data (e.g., Annuli dataset with 50 points)
2. Toggle "Show Betti Numbers" to ON
3. Click "Train 1 Step" or "Train 100 Steps"
4. Expected: After each training step, if the poset structure changes, Betti numbers update automatically

**Test: Decision Boundary Changes**
1. With Betti numbers toggle ON, pan or zoom the decision boundary plot
2. Expected: If new regions appear or disappear, Betti numbers update

### 3. Performance and Caching

**Test: No Redundant Computation**
1. Toggle Betti numbers ON
2. Click "Forward Pass" multiple times with the same inputs
3. Expected: Betti numbers should only compute once (first time)
4. Check browser console for computation logs
5. Should see "Betti numbers unchanged, skipping computation" (or no computation)

**Test: Change Detection**
1. Toggle Betti numbers ON with a simple network
2. Note the current Betti numbers
3. Randomize weights
4. Click "Forward Pass"
5. Expected: If the topology changes, Betti numbers recompute
6. If topology is unchanged, should use cached result

### 4. Edge Cases

**Test: No Nerve Complex Data**
1. Toggle Betti numbers ON before switching to Nerve Complex view
2. Expected: Should show an appropriate error message
3. Switch to Nerve Complex view
4. Expected: Betti numbers should then compute successfully

**Test: Empty Network**
1. Clear all training data
2. Reset network to initial state
3. Toggle Betti numbers ON in Nerve Complex view
4. Expected: Should handle gracefully, possibly showing "No simplices" message

**Test: Toggle Off During Computation**
1. Use a complex network that takes time to compute
2. Toggle Betti numbers ON
3. Immediately toggle OFF before computation finishes
4. Expected: Display should clear, computation may continue in background but results won't display

### 5. Integration Testing

**Test: Dual Graph to Nerve Complex Switch**
1. Start in Dual Graph view
2. Toggle to Nerve Complex view
3. Toggle Betti numbers ON
4. Expected: Betti numbers compute for the current nerve complex
5. Toggle back to Dual Graph view
6. Expected: Betti numbers section should hide (as per existing behavior)
7. Toggle back to Nerve Complex view
8. Expected: Betti numbers reappear (if toggle is still ON)

### 6. Visual Verification

**Test: Display Format**
1. With Betti numbers displayed, verify:
   - β₀, β₁, β₂, etc. are shown with subscript formatting
   - Each has an interpretation (connected components, loops, voids, etc.)
   - Poset statistics shown at bottom (elements, edges, dimension)
   - Loading state shows "Computing..." in gray italic

**Test: Error States**
1. Force an error (e.g., by modifying code temporarily)
2. Expected: Error message in red color
3. User-friendly error text

## Expected Behavior Summary

### When Toggle is ON:
- Betti numbers compute immediately when toggled
- Auto-update whenever poset structure changes
- Show "Computing..." message during computation
- Cache results to avoid redundant computation
- Display results with proper formatting

### When Toggle is OFF:
- Display area is empty
- No computation occurs
- No updates happen even if poset changes
- State is saved (toggle remembers it was off)

### Change Detection:
- Compares number of simplices
- Compares edge structure
- Compares individual simplices (order-independent)
- Only recomputes if actual structural change detected

## Known Limitations

1. **Performance**: Very complex networks (many neurons, many regions) may have slow computation
2. **Deep Copy**: Uses JSON serialization for deep copying (may be slow for large posets)
3. **Comparison**: Edge comparison uses stringified arrays (could be optimized)

## Success Criteria

All tests should pass with:
- ✓ Toggle works immediately
- ✓ Auto-update triggers on poset changes
- ✓ No redundant computation when poset unchanged
- ✓ Proper error handling
- ✓ Smooth UI updates
- ✓ Correct formatting and display
- ✓ Integration with nerve complex rendering works seamlessly

## Debugging Tips

If issues occur:
1. Open browser console to see computation logs
2. Check for errors in red text
3. Verify nerve complex is rendering properly first
4. Ensure network has at least 2 inputs for decision boundary
5. Check that poset has valid structure (simplices and edges arrays)
