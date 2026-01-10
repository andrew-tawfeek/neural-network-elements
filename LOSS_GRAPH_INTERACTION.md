# Loss Graph Interactive Navigation

## Overview

The loss graph now features intelligent interaction controls that change based on training state. Users can drag through the loss history to jump to any training epoch when training is paused, but interaction is blocked during active training.

## Features

### 1. Training State-Aware Interaction

**When Training is Active (Play button pressed):**
- All mouse interactions are blocked
- Cursor shows "not-allowed" icon
- Instruction text displays in red: "⏸ Pause training to navigate the loss graph"
- Graph continues to update with new loss values
- Prevents accidental navigation during training

**When Training is Paused:**
- Full interactive drag functionality enabled
- Cursor shows "grab" icon when hovering over graph
- Instruction text displays: "Drag to jump to any training epoch"
- User can explore training history freely

### 2. Click-and-Drag Navigation

**How it Works:**
1. **Mouse Down**: Start dragging anywhere on the loss graph
2. **Mouse Move**: Drag left/right to navigate through training epochs
3. **Mouse Up**: Release to finalize position and see notification
4. **Visual Feedback**: Current epoch highlighted with red marker and vertical dashed line

**During Drag:**
- Network weights/biases are restored to selected epoch
- All visualizations update in real-time:
  - Network visualization
  - Decision boundary
  - Binary display
  - 3D plots (if active)
- Loss graph shows current position with highlight
- Stats display updates to show current step

**After Drag:**
- Popup notification shows:
  - Step number jumped to
  - Loss value at that step
- Weight/bias controls reflect historical state
- All visualizations match selected epoch

### 3. Visual Indicators

**Current Step Highlight:**
- Large red circle with white center
- Vertical dashed line through the graph
- Stats panel shows "Viewing: Step N" in red (when not at latest)

**Cursor States:**
- `not-allowed`: Training is active (can't interact)
- `grab`: Hovering over graph (training paused)
- `grabbing`: Actively dragging
- `default`: Outside interactive area

**Instruction Text:**
- Changes color and message based on training state
- Red text when blocked (training active)
- Gray text when enabled (training paused)

### 4. Network State Management

**State Storage:**
- Each training step saves complete network state
- Includes all weights, biases, and activations
- Stored in `networkStates` array alongside loss history
- Limited to `maxDataPoints` (1000) recent steps

**State Restoration:**
- Instantly restores network to any historical epoch
- Calls `window.net.loadState()` with saved state
- Updates all UI components to match

## Implementation Details

### New Properties (js/loss-graph.js)

```javascript
isDragging: false,        // Currently dragging?
dragStartX: 0,            // Starting X position of drag
lastDragStep: -1,         // Last step jumped to during drag
```

### Key Methods

**`isTrainingActive()`**
- Checks `TrainingManager.isContinuousTraining`
- Returns true if Play button is active
- Used to block/enable interactions

**`handleMouseDown(event)`**
- Blocks if training active
- Starts drag operation
- Jumps to initial click position

**`handleMouseMove(event)`**
- Shows "not-allowed" cursor if training active
- Updates position during drag
- Shows "grab" cursor when hovering (paused)

**`handleMouseUp(event)`**
- Ends drag operation
- Shows notification popup
- Resets cursor

**`handleMouseLeave(event)`**
- Cancels drag if mouse leaves canvas
- Prevents stuck drag state

**`jumpToStepDuringDrag(step)`**
- Optimized for frequent updates during drag
- Only updates essential visualizations
- Skips expensive operations like `createWeights()`
- Updates loss graph and binary display only

### Integration Points

**Training Manager Integration:**
- `startContinuousTraining()` - Redraws graph with blocked state
- `stopContinuousTraining()` - Redraws graph with enabled state
- Graph instruction text updates automatically

**Visualization Updates:**
- All visualizations update when jumping to epochs
- Throttled during drag for performance
- Full update on mouse up

## User Experience

### Workflow Example

1. **Train the network**: Click Play or Train 100 Steps
2. **Watch loss decrease**: Graph updates in real-time
3. **Pause training**: Click Pause button
4. **Explore history**: Drag on graph to see network at different epochs
5. **Compare states**: See how decision boundaries evolved
6. **Resume training**: Click Play to continue from current epoch
7. **Jump to best**: Drag to minimum loss point to restore best weights

### Use Cases

**Debugging Training:**
- Jump to epoch where loss started increasing (overfitting)
- Compare network behavior before/after learning plateau
- Find optimal stopping point

**Understanding Learning:**
- Watch decision boundary evolution by dragging through epochs
- See how activation patterns change over time
- Identify when network learned specific patterns

**Recovering Best Model:**
- Drag to lowest loss point
- Save network state at that epoch
- Avoid overfitting to recent data

**Presentation/Demo:**
- Smoothly animate training by dragging
- Show specific interesting epochs
- Compare different training stages

## Performance Optimizations

### During Drag
- Skips `createWeights()` (expensive DOM manipulation)
- Updates only essential visualizations
- Uses `requestAnimationFrame` timing
- Prevents redundant updates to same step

### During Training
- All interactions completely blocked
- No event processing overhead
- Graph updates only when new loss values arrive
- Instruction text cached between redraws

## Backward Compatibility

- Old click-based navigation removed
- All existing loss graph features preserved
- Network state save/load still works
- Stats display unchanged
- Keyboard navigation not affected

## Technical Notes

### Margin Calculations
```javascript
const margin = { left: 60, right: 30, top: 20, bottom: 50 };
```
- Consistent across all mouse event handlers
- Matches drawing margin for accurate positioning

### Coordinate Conversion
```javascript
const relativeX = (x - margin.left) / plotWidth;
const targetStep = Math.round(relativeX * (this.lossHistory.length - 1));
```
- Converts mouse X to step index
- Clamps to valid range [0, 1]
- Rounds to nearest step

### State Tracking
- `isDragging`: Prevents cursor flickering
- `lastDragStep`: Avoids redundant updates
- `currentStep`: Always tracks viewed epoch
- Cleared on mouse up/leave

## Future Enhancements

Potential improvements:
1. **Pinch zoom**: Navigate specific epoch ranges
2. **Keyboard shortcuts**: Arrow keys to step through epochs
3. **Bookmarks**: Mark interesting epochs for quick access
4. **Diff view**: Compare two epochs side-by-side
5. **Animation playback**: Auto-play through training history
6. **Epoch selection**: Select range of epochs for analysis

## Testing Recommendations

1. **Basic Drag**:
   - Generate training data
   - Train for 50-100 steps
   - Pause training
   - Drag across graph
   - Verify network state changes

2. **Training Block**:
   - Start continuous training
   - Try to drag graph
   - Verify "not-allowed" cursor
   - Verify no interaction possible

3. **State Restoration**:
   - Train network
   - Note decision boundary at step 50
   - Train more (to step 100)
   - Drag back to step 50
   - Verify boundary matches original

4. **Pause/Resume**:
   - Start training
   - Pause
   - Drag to middle epoch
   - Resume training
   - Verify continues from that epoch

5. **Edge Cases**:
   - Drag with no data (should not crash)
   - Drag with 1 data point (should highlight)
   - Drag outside graph area (should not jump)
   - Mouse leave during drag (should cancel)

## Conclusion

The interactive loss graph provides an intuitive way to explore training history while preventing accidental interference during active training. The drag-based navigation makes it easy to compare network states across different epochs, helping users understand how their network learned and identify optimal checkpoints.
