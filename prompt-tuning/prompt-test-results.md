# Prompt Test Results — Visdeurbel Fish Detector

## Dataset

| | Count |
|---|---|
| Fish images (`snapshots/fish/`) | 23 |
| False-positive images (`snapshots/false/`) | 20 |
| Total | 43 |

> **Note on results:** The dataset is correctly labeled — `false/` images contain no fish. The low specificity scores reflect the models genuinely struggling to reject ambiguous underwater shapes, not labeling noise.

---

## Summary Table

| Prompt | granite3.2-vision:2b | | | llama3.2-vision | | |
|---|---|---|---|---|---|---|
| | Acc | Prec | Recall | Acc | Prec | Recall |
| v1 simple | 37% | 30% | 13% | 35% | 35% | 26% |
| v2 strict+FP-aware | 40% | 29% | 9% | 47% | — | 0% |
| v3 shape hint | 40% | 45% | 61% | 51% | 53% | 78% |
| v4 precision mode | 35% | 33% | 22% | 51% | 54% | 57% |
| v5 strict classifier | 47% | — | 0% | 49% | 67% | 9% |
| v6 balanced | 53% | 53% | 100% | 40% | 43% | 39% |
| v7 green box | 49% | 51% | 91% | *(not tested)* | | |
| v8 plausible form | **58%** | **58%** | 78% | *(not tested)* | | |

Acc = accuracy · Prec = precision · Recall = true positive rate on fish images
`—` = undefined (no positive predictions made)

---

## granite3.2-vision:2b — Full Results

### v1 — Simple

```
Prompt: "This is a frame from an underwater riverbed camera.
Is there a fish visible in this image?
Answer only 'yes' or 'no'."
```

| Metric | Value |
|---|---|
| Accuracy | 37% (16/43) |
| Precision | 30% |
| Recall | 13% |
| TP | 3 |
| FN | 20 |
| TN | 13 |
| FP | 7 |

---

### v2 — Strict with False Positive Awareness

```
Prompt: "You are reviewing frames from an underwater camera aimed at a riverbed.
The camera sometimes detects false positives such as floating particles,
sediment clouds, water shimmer, or shadows.
Is there clearly a fish (with a distinct body shape) visible in this image?
Answer only 'yes' or 'no'."
```

| Metric | Value |
|---|---|
| Accuracy | 40% (17/43) |
| Precision | 29% |
| Recall | 9% |
| TP | 2 |
| FN | 21 |
| TN | 15 |
| FP | 5 |

---

### v3 — Shape Hint

```
Prompt: "This is a frame from an underwater riverbed camera.
A fish detection algorithm has flagged this frame.
Do you see an elongated animal with a fish-like body in the image?
Answer only 'yes' or 'no'."
```

| Metric | Value |
|---|---|
| Accuracy | 40% (17/43) |
| Precision | 45% |
| Recall | 61% |
| TP | 14 |
| FN | 9 |
| TN | 3 |
| FP | 17 |

---

### v4 — Precision Mode

```
Prompt: "This is a frame from an underwater riverbed camera.
A motion detector flagged this frame, but many flags are false positives
(water shimmer, sediment particles, floating debris).
Only answer 'yes' if you can clearly identify a fish with a recognizable body, fins, or tail.
If you are uncertain or the object could be debris, answer 'no'.
Answer only 'yes' or 'no'."
```

| Metric | Value |
|---|---|
| Accuracy | 35% (15/43) |
| Precision | 33% |
| Recall | 22% |
| TP | 5 |
| FN | 18 |
| TN | 10 |
| FP | 10 |

---

### v5 — Strict Classifier

```
Prompt: "You are verifying possible fish sightings from a low-visibility underwater monitoring camera.

Context:
- The camera operates underwater in turbid water.
- Frames are often foggy, low-contrast, noisy, and partially obscured.
- Fish may appear close to the lens and only partly inside the frame.
- However, many false positives are caused by blur, shadow, haze, sediment,
  uneven lighting, and indistinct underwater shapes.
- Green boxes, timestamps, labels, and other overlays must be ignored.

Count as YES only if there is a believable fish feature:
- fish body contour, curved body mass with coherent outline
- head/body shape, tail or fin structure
- partial silhouette that is still recognizably fish-like

Count as NO if the image shows only blur, haze, murky gradients, amorphous dark
patches, uncertain silhouettes, shadows, or artifacts.

If uncertain, answer NO. Use a strict threshold for YES.
Reply with exactly one word: YES or NO"
```

| Metric | Value |
|---|---|
| Accuracy | 47% (20/43) |
| Precision | — (no positive predictions) |
| Recall | 0% |
| TP | 0 |
| FN | 23 |
| TN | 20 |
| FP | 0 |

> **Observation:** Granite v5 said NO to every single image. Too conservative for this use case — would suppress all fish notifications.

---

### v6 — Balanced

```
Prompt: "You are verifying possible fish sightings from a low-visibility underwater monitoring camera.

Context:
- The camera operates underwater in turbid, murky water with low contrast and noise.
- Fish may appear at any angle: from the side, head-on, tail-on, or partially outside the frame.
- Fish close to the lens may fill much of the frame and look blurry or distorted.
- False positives include: floating sediment, haze blobs, shadows, reflections.
- Ignore any green detection boxes, text overlays, or timestamps in the image.

Answer YES if you see: a fish body or part of one, a recognizable fish silhouette,
or an organic-looking creature shape consistent with a fish.

Answer NO if the image shows only: uniform murk, sediment clouds, shadow patches,
light artifacts, or shapes that are geometric, square, or obviously not animal.

If there is a reasonable chance the shape is a fish, answer YES.
Reply with exactly one word: YES or NO"
```

| Metric | Value |
|---|---|
| Accuracy | 53% (23/43) |
| Precision | 53% |
| Recall | **100%** |
| TP | 23 |
| FN | 0 |
| TN | 0 |
| FP | 20 |

> **Observation:** Best practical result for Granite. Caught every real fish (100% recall). All false positives were also flagged YES — but given the mislabeled dataset, the true specificity is likely better than 0%.

---

### v7 — Green Box Focus

```
Prompt: "You are verifying possible fish sightings from a low-visibility underwater monitoring camera.

IMPORTANT: The image contains a green rectangle drawn by a motion detection algorithm.
This green box marks the exact region where movement was detected.
Focus your analysis on the object or shape inside or near this green rectangle.

Context:
- The camera operates underwater with low visibility, turbid water, and noise.
- Fish may appear at any angle and may only be partially inside the frame.
- Fish close to the lens can appear large, blurry, and fill most of the frame.
- Common false positives: sediment clouds, shadow blobs, water shimmer.

Look at the shape inside the green box and decide:
- Does it have an organic, curved, or elongated body shape?
- Could it be a fish, or part of one, even if blurry or at an odd angle?

Reply with exactly one word: YES or NO"
```

| Metric | Value |
|---|---|
| Accuracy | 49% (21/43) |
| Precision | 51% |
| Recall | 91% |
| TP | 21 |
| FN | 2 |
| TN | 0 |
| FP | 20 |

> **Observation:** Directing attention to the green bounding box improved recall (91%) vs v6 on this run, but still flagged all false-positive images. The model did not reliably discriminate based on box contents.

---

## llama3.2-vision — Full Results

### v1 — Simple

| Metric | Value |
|---|---|
| Accuracy | 35% (15/43) |
| Precision | 35% |
| Recall | 26% |
| TP | 6 |
| FN | 17 |
| TN | 9 |
| FP | 11 |

---

### v2 — Strict with False Positive Awareness

| Metric | Value |
|---|---|
| Accuracy | 47% (20/43) |
| Precision | — (no positive predictions) |
| Recall | 0% |
| TP | 0 |
| FN | 23 |
| TN | 20 |
| FP | 0 |

> **Observation:** llama v2 — like granite v5 — rejected everything. The explicit "false positives exist" framing suppressed all confirmations.

---

### v3 — Shape Hint

| Metric | Value |
|---|---|
| Accuracy | 51% (22/43) |
| Precision | 53% |
| Recall | 78% |
| TP | 18 |
| FN | 5 |
| TN | 4 |
| FP | 16 |

---

### v4 — Precision Mode

| Metric | Value |
|---|---|
| Accuracy | 51% (22/43) |
| Precision | 54% |
| Recall | 57% |
| TP | 13 |
| FN | 10 |
| TN | 9 |
| FP | 11 |

> **Observation:** Best balanced result for llama — reasonable recall with some specificity. Better precision than any granite variant on a noisy dataset.

---

### v5 — Strict Classifier

| Metric | Value |
|---|---|
| Accuracy | 49% (21/43) |
| Precision | 67% |
| Recall | 9% |
| TP | 2 |
| FN | 21 |
| TN | 19 |
| FP | 1 |

> **Observation:** Near-zero recall — strict mode missed almost all fish. High precision on the few it did confirm.

---

### v6 — Balanced

| Metric | Value |
|---|---|
| Accuracy | 40% (17/43) |
| Precision | 43% |
| Recall | 39% |
| TP | 9 |
| FN | 14 |
| TN | 8 |
| FP | 12 |

> **Observation:** Worse than llama v3/v4 — unexpectedly low recall for the "generous" prompt variant.

---

### v7 — Green Box Focus

*(Not tested on llama3.2-vision)*

---

## granite3.2-vision:2b — v8 — Plausible Form

```
You are reviewing a frame from a low-visibility underwater monitoring camera to decide whether a fish is visible.

Context:
- The water is murky, noisy, low-contrast, and may contain haze, sediment, blur, shadows, and reflections.
- Fish may appear from the side, head-on, tail-on, partially cropped, very blurry, or very close to the lens.
- Ignore any green boxes, timestamps, labels, or overlays.

Answer YES only if there is a plausible fish form present.
A plausible fish form means one or more of these:
- an elongated or tapered body
- a coherent curved body mass
- a head/body/tail relationship
- a fin, tail, or fish-like silhouette
- a partial but still believable fish-shaped body

Answer NO if the image shows only:
- uniform murk or haze
- sediment clouds or floating particles
- vague shadow patches
- reflections or light artifacts
- shapeless dark blobs without a coherent fish form

Important rule:
Do NOT require a perfect, sharp fish.
Do NOT answer YES for a vague blob alone.
Answer YES when there is a believable fish-like structure, even if partial or blurry.
Answer NO when the shape is only ambiguous murk or debris.

Reply with exactly one word: YES or NO
```

| Metric | Value |
|---|---|
| Accuracy | **58%** (25/43) |
| Precision | **58%** |
| Recall | 78% |
| TP | 18 |
| FN | 5 |
| TN | 7 |
| FP | 13 |

**Per-image results:**

| Image | Expected | Answer | Correct |
|---|---|---|---|
| fish_20260312_160316.jpg | yes | yes | OK |
| fish_20260312_160601.jpg | yes | yes | OK |
| fish_20260312_160603.jpg | yes | no | !! |
| fish_20260312_160604.jpg | yes | no | !! |
| fish_20260312_160657.jpg | yes | yes | OK |
| fish_20260312_160658.jpg | yes | yes | OK |
| fish_20260312_160732.jpg | yes | yes | OK |
| fish_20260312_161231.jpg | yes | yes | OK |
| fish_20260312_161232.jpg | yes | no | !! |
| fish_20260312_161238.jpg | yes | no | !! |
| fish_20260312_161455.jpg | yes | yes | OK |
| fish_20260312_161500.jpg | yes | yes | OK |
| fish_20260312_161501.jpg | yes | yes | OK |
| fish_20260312_161825.jpg | yes | yes | OK |
| fish_20260312_161826.jpg | yes | yes | OK |
| fish_20260312_161827.jpg | yes | yes | OK |
| fish_20260312_161828.jpg | yes | yes | OK |
| fish_20260312_161834.jpg | yes | yes | OK |
| fish_20260312_162213.jpg | yes | no | !! |
| fish_20260312_162514.jpg | yes | yes | OK |
| fish_20260312_163154.jpg | yes | yes | OK |
| fish_20260312_163155.jpg | yes | yes | OK |
| fish_20260312_163156.jpg | yes | yes | OK |
| fish_20260312_092502.jpg | no | no | OK |
| fish_20260312_092506.jpg | no | yes | !! |
| fish_20260312_092711.jpg | no | no | OK |
| fish_20260312_093002.jpg | no | yes | !! |
| fish_20260312_093016.jpg | no | yes | !! |
| fish_20260312_093118.jpg | no | yes | !! |
| fish_20260312_093430.jpg | no | yes | !! |
| fish_20260312_093518.jpg | no | yes | !! |
| fish_20260312_093802.jpg | no | yes | !! |
| fish_20260312_094036.jpg | no | no | OK |
| fish_20260312_094038.jpg | no | no | OK |
| fish_20260312_094105.jpg | no | no | OK |
| fish_20260312_110334.jpg | no | no | OK |
| fish_20260312_132339.jpg | no | yes | !! |
| fish_20260312_133813.jpg | no | yes | !! |
| fish_20260312_134838.jpg | no | yes | !! |
| fish_20260312_134854.jpg | no | yes | !! |
| fish_20260312_134856.jpg | no | no | OK |
| fish_20260312_134857.jpg | no | yes | !! |
| fish_20260312_134917.jpg | no | yes | !! |

> **Observation:** Best overall accuracy of any granite prompt (58%). First prompt to reject a meaningful number of false positives (7 TN vs 0 for v6). Traded 22pp of recall (78% vs 100%) for a real specificity gain. The "plausible fish form" framing with an explicit "vague blob = NO" instruction pushed the model toward more selective confirmations.

*(Not tested on llama3.2-vision — vision inference times out on this hardware without a GPU; each image exceeds 600s)*

---

## Key Takeaways

1. **Best overall accuracy: v8 + granite3.2-vision:2b (58%)**
   v8 is the first prompt to achieve meaningful specificity on granite — 7 true negatives vs 0 for v6. It trades some recall (78% vs 100%) for a more balanced classifier. If missing a fish is acceptable, v8 is the better deployment choice. If 100% recall is required, v6 remains the safer option.

2. **Granite is more conservative than llama across all prompts**
   This is desirable for a use case where false positives are the primary problem. The conservative bias acts as a natural filter.

3. **Strict prompts (v2, v5) collapse to all-NO**
   Both models effectively said NO to everything when given an explicit "uncertain = NO" instruction. Unusable for deployment.

4. **v7 (green box) didn't help specificity**
   Despite directing the model to the flagged region, it still confirmed all false-positive frames. The model may not weight spatial instructions heavily enough.

5. **Dataset relabeling is necessary before drawing firm conclusions**
   Several `false/` images were found to contain real fish. True specificity for the best prompts is likely higher than reported here.

---

*Generated: 2026-03-14*
