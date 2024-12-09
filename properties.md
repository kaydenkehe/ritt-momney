any good counterfactual sequence should have the following properties (from easiest to hardest to measure):<br>
1. output linearity
    - regressor outputs should have a constant rate of change (which implies monotonicity).
    - justification: this is a restatement of the goal, to generate cf images that produce a continuous directional change in the output.
    - approach: just compute the r2 value.
2. continuity
    - each successive cf should be a small step from the previous, and all step sizes should be similar
    - justification: this is also a restatement of the goal. the larger the change between successive cfs, the less we get to see *how* the input has to change to achieve our desired output.
    - approach: we can again just compute the r2 value on some metric measuring the distance between successive cfs.
        - difference metrics: absolute difference, MSE, PSNR, SSIM
3. plausibility
    - each frame of the stream should be in the input space of the model
    - justification: if cfs aren't in the model's input space, the model behavior is irrelevant.
    - approach: this is a bit harder. we could train a classifier to predict whether a given image is in the input space of the model and aggregate the predictions for each frame. we could also operate under the assumption that the original input is in the input space and use some measure of distance between each cf and the input (SSIM again?).
4. locality -- debating removing this one
    - changes should be localized to specific, relevant parts of the input
    - justification: in the case of something like CXR-Age where the model's decision-making process is highly complex and multi-faceted, we gain more useful information from observing many independent local changes than one global series of changes. interactions between different simultaneous changes are hard to reason about.
    - approach: like, who even cares.
5. structure preservation
    - the essential semantic structure of the input image should be preserved
    - justification: by anchoring the stream to some fundamental structure, it becomes easier to interpret and isolate changes relevant to the model. we don't want to see how some non-specific image has to change to achieve an output, we want to see how an *input* has to change. in the case of CXR-Age for example, it should be clear that despite their differences, the frames in a stream of counterfactuals are all x-rays of the same patient.
    - approach: this is extremely difficult. any notion of 'essential semantic structure' is subjective and domain-specific. this could lend itself to a separate model for evaluating the extent to which two images share an underlying strucutre in a domain. that's clearly a cop-out, though. seems totally impracticable.

<br><br>
aggregation:<br>
- it'll be most helpful to evaluate each metric individually, but it'd be nice to have a single aggregate metric.
- it seems plausible that each of our individual metrics will be between 0 and 1, which is helpful.
- geometric mean seems to be a winner. it has some desirable properties - any individual low value is a pretty severe penalty, but all high values will result in a high aggregate (unlike just multiplying the values together, for example)

<br><br>
additional thoughts:<br>
- whether a change in cfs is 'small' seems domain-specific, but i'm unsure and open to being persuaded by empirical results.
- debating making intra-difference smoothness a desirable metric (inter-difference smoothness we already have). this feels like it might be domain-specific. in a stupid degenerate case, what if we were generating counterfactuals for a regressor that predicted how noisy an image was? then, intra-difference smoothness would be undesirable.
- debating including locality for the same reason as above, and also because i haven't convinced myself that it's actually *helpful*.

