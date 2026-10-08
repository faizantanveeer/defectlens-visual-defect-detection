# Video walkthrough script (2–3 minutes)

Target: about 2:45. Screen layout: README on the left, terminal on the right. Rough timings in brackets.

---

### 1. Problem and result [0:00–0:20]

> "A production line inspects parts by hand. I built a system that classifies each image as normal or
> defective and serves it as an API. The dataset wasn't attached, so I used MVTec AD metal nut: real
> industrial photos, four defect types, 28% defective. The pipeline is dataset-agnostic, so the
> client's data drops in with no code changes."

*Show:* README header table.

### 2. Dataset decisions [0:20–0:50]

> "Three decisions came from looking at the data. First, I stratified the split by **defect type**,
> not just class, so every defect type appears in every split. Second, **no flip augmentation**: the
> nut is chiral, so a mirrored good nut looks exactly like the 'flip' defect. Flipping would have
> poisoned the labels. Third, no hue jitter, because it would erase the 'color' defect.
> I also caught a bug in my own duplicate check: a 64-bit perceptual hash merged 299 of 335 images
> into one group, because identical-looking parts are only a few bits apart. I fixed it, and the
> split now refuses to run if that ever happens again."

*Show:* `reports/eda/samples_per_type.png`, the augmentation table.

### 3. Model and training [0:50–1:20]

> "Transfer learning, because 217 training images is far too few to train from scratch. I compared four
> ImageNet backbones on the identical split. All rank the test set perfectly, so I chose on latency:
> **MobileNetV3 runs in 5 ms on CPU**, 2× faster than EfficientNet and 9× faster than ResNet-50, and the larger
> ResNets each missed a defect. Imbalance is handled with a weighted loss, model selection on average
> precision, and an F2 threshold, because a missed defect costs more than a re-inspection.
> Two training bugs: validation collapsed when fine-tuning started, a BatchNorm train/eval mismatch,
> fixed by freezing BN statistics; and a loss spike, fixed with LR warm-up."

*Show:* model comparison table, training curves.

### 4. Evaluation and error analysis [1:20–2:05]

> "The held-out test is perfect, 19 of 19 defects and 48 of 48 good parts. But 67 images can't prove
> a perfect model; the exact confidence interval still allows recall down to 82%. So I ran 5-fold
> cross-validation, where every image is scored by a model that never saw it: **97.8% of defects caught,
> 99.6% of good parts passed**, three errors in 335 images. One of those 'false positives' is a good-labelled
> nut with visible scratch marks: likely label noise.
> Cross-validation also exposed a weakness: on a small validation set, one hard example dragged the
> threshold to nearly zero and caused false rejects. Bounding the threshold search fixed that.
> Grad-CAM shows the model looks at the actual defect: the attention peak is inside the defect mask
> for 84% of detections. The misses are thin, low-contrast scratches, a resolution limit at 224 pixels.
> Under perturbations it's robust to blur, noise and JPEG; −30% lighting is the sensitive axis.
> Finally, I tested a defect type the model has never seen, by retraining without it: recall drops
> from 98% to about 58%, though those parts still score above good ones. That's the key limitation."

*Show:* CV report, Grad-CAM gallery, robustness table.

### 5. Production service [2:05–2:40]

*Live:* `docker run -p 8000:8000 defect-detection-api` → `/docs` → upload `defective_scratch.png` → response.
Then upload a text file → 415 error. Then `curl localhost:8000/metrics`.

> "Serving uses ONNX Runtime with no PyTorch in the image. Evaluation ran through the same predictor
> class as the API, so the reported numbers are the deployed model's numbers. The response gives the
> class, the confidence, and a **review flag** for borderline scores, so uncertain parts go to a human.
> Inputs are validated (type, size, corrupt files, decompression bombs). Logs are JSON with request
> IDs, Prometheus metrics include the score distribution as a drift signal, and overload returns 503
> instead of growing latency. It's stateless and scales horizontally; there's a Kubernetes manifest with an autoscaler."

### 6. Limitations [2:40–2:55]

> "The main limitation: a supervised classifier only knows defects it has seen. In production I'd run
> an unsupervised anomaly detector such as PatchCore alongside it, raise resolution or tile the image for
> hairline scratches, and recalibrate on the client's real line data."
