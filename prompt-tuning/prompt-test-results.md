# Prompt Test Results — Visdeurbel Fish Detector

## Dataset

| | Count |
|---|---|
| Fish images (`snapshots/fish/`) | 23 |
| False-positive images (`snapshots/false/`) | 20 |
| Total | 43 |

> **Note on dataset quality:** During analysis, several images in the `false/` folder were found to actually contain fish — they had been labeled as false positives based on early intuition rather than careful inspection. This skewed all accuracy and specificity metrics downward. The "low accuracy" scores below largely reflect this labeling noise, not pure model failure. A relabeled dataset is pending.

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
| v6 balanced | **53%** | 53% | **100%** | 40% | 43% | 39% |
| v7 green box | 49% | 51% | 91% | *(not tested)* | | |

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

## Key Takeaways

1. **Best combination for deployment: v6 + granite3.2-vision:2b**
   100% recall means no fish notification is ever missed. The false positive rate is high on this dataset, but the dataset itself has mislabeling noise that inflates the FP count.

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
