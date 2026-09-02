# Numerical Range and Precision Analysis of DNN Data (FP32 vs FP16 / FP8 / INT16 / INT8)

Range and precision study of the data consumed and produced by the layers of a deep
neural network during **training** and **inference**, and an assessment of whether
reduced-precision representations can replace FP32.

Course assignment submission.

---

## 1. Experiments

| analysis  | model       | dataset   | representation | data source |
|-----------|-------------|-----------|----------------|-------------|
| training  | ResNet-18   | CIFAR-10  | FP32           | Kaggle      |
| inference | ResNet-18   | CIFAR-10  | FP32           | Kaggle      |
| inference | MobileNetV2 | CIFAR-100 | FP32           | Kaggle      |
| inference | LeNet-5     | MNIST     | FP32           | Kaggle      |

**Training analysis.** ResNet-18 is trained on CIFAR-10 in FP32 and instrumented at
three stages — `train_start` (first mini-batches of epoch 1), `train_middle`
(epoch E/2) and `train_end` (last epoch). At each stage the following are recorded
**separately for every layer**:

* inputs (the normalised images entering the network)
* layer inputs (the operand entering each individual layer)
* weights
* bias values, including BatchNorm gamma and beta
* activations (layer outputs)
* gradients with respect to activations (the back-propagated error)
* gradients with respect to weights
* weight updates (the delta actually applied by the optimiser)

**Inference analysis.** Each of the three models performs one complete FP32 pass
over its full test set, recording per layer: inputs, weights, biases, activations
and output logits. Weights and biases are read once; activations and logits are
accumulated over the entire test set.

**Layer coverage.** Convolution, linear, batch-normalisation, activation
(ReLU / ReLU6) and pooling (max, average, adaptive average) layers are all
instrumented via forward and backward hooks.

---

## 2. Method

### Statistics collection

All statistics are gathered in a **single streaming pass**, so a full test-set
sweep never has to be held in memory.

Computed **exactly**, over every value of every tensor:

* count, number of zeros, number of negatives, non-finite count
* minimum and maximum value
* smallest non-zero magnitude and largest magnitude
* mean, standard deviation, mean absolute value
* the complete histogram of `floor(log2|x|)` — one bin per binade. This is the
  dynamic-range profile, and it is what determines directly whether a format's
  exponent field is wide enough.

Computed from a bounded **reservoir sample** per tensor (used for the density
plots and the error metrics):

* number of distinct values observed
* smallest and median spacing between distinct observed values
* quantisation error against each candidate format

### Reduced-precision assessment

No low-precision training or inference is performed — as the assignment permits,
the alternative representations are *simulated* on the captured FP32 data.

* **FP16, BF16, FP8-E4M3, FP8-E5M2** — every value is rounded to the nearest
  representable number of the target format (round-to-nearest-even), with gradual
  underflow through subnormals and saturation at the format maximum. This yields
  the measured SQNR, the fraction of values that would **overflow**, the fraction
  that would **flush to zero**, and the fraction landing in the subnormal range.
* **INT16 and INT8** — symmetric per-tensor uniform quantisation,
  `q = clamp(round(x / scale), -(2^(b-1)-1), 2^(b-1)-1)`, evaluated with two
  scaling strategies:
  * `scale = max|x| / (2^(b-1)-1)` — no clipping, worst resolution
  * `scale = P99.9(|x|) / (2^(b-1)-1)` — clips the tail, the usual post-training
    quantisation choice
  reporting SQNR, effective number of bits, the clipped fraction and the fraction
  of non-zero values driven to zero.

### Verdict thresholds

| verdict | condition |
|---|---|
| OK | SQNR >= 40 dB (about 6.4 effective bits preserved) |
| MARGINAL | SQNR between 25 and 40 dB |
| INSUFFICIENT | SQNR < 25 dB |
| OVERFLOW | some values exceed the format's largest finite value |
| UNDERFLOW | more than 1% of non-zero values flush to zero |
| RISKY | an INT scale that clips more than 1% of the values |

---

## 3. Repository contents

```
dnn_precision_analysis.py        complete implementation, single file
DNN_Precision_Analysis.ipynb     Colab notebook that runs it end to end
results/
    REPORT.md                    measured ranges, verdicts and conclusions
    figures/
        <phase>/<datatype>.png       three panels: full numerical range |
                                     magnified around zero | magnitude
                                     histogram with FP16 / FP8 / INT8 limits
        <phase>/per_layer/*.png      per-layer plots annotated with min, max,
                                     smallest non-zero magnitude and spacing
        <phase>/_range_overview.png  magnitude span of every data type on one axis
    tables/
        summary_all_tensors.csv      per layer: min, max, smallest non-zero
                                     magnitude, mean, std, zero fraction,
                                     dynamic range in binades and dB, distinct
                                     values, observed value spacing, FP32 ULP
        format_feasibility.csv       per layer per format: SQNR, ENOB, overflow,
                                     flush-to-zero, subnormal, clipping and
                                     zeroed fractions, scale, max error, verdict
```

`<phase>` is one of `train_start`, `train_middle`, `train_end`,
`infer_resnet18_cifar10`, `infer_mobilenetv2_cifar100`, `infer_lenet5_mnist`.

---

## 4. Reading the plots

Every figure has three panels:

1. **Full range** — normalised histogram (probability density) of the values on the
   real-number line, with a logarithmic y-axis so the tails remain visible.
2. **Magnified around zero** — the same data restricted to a narrow window around
   zero, where the small and closely spaced values live.
3. **Magnitude versus format limits** — histogram of `floor(log2|x|)`. Dashed
   vertical lines mark each format's smallest subnormal, dotted lines its largest
   finite value, and the shaded bands show the roughly 7-binade (INT8) and
   15-binade (INT16) windows reachable with a single per-tensor scale factor.
   Anything to the left of a format's dashed line flushes to zero; anything to the
   right of its dotted line overflows.

---

## 5. Findings

The measured numbers, the per-layer verdicts and the discussion are in
[`results/REPORT.md`](results/REPORT.md), with the raw values in the two CSV files
under `results/tables/`.

---

## 6. Reproducing

Open `DNN_Precision_Analysis.ipynb` in Google Colab, select a T4 GPU runtime, put a
Kaggle API token in Step 2 and run all cells. From a shell:

```bash
pip install torch torchvision kagglehub matplotlib numpy
python dnn_precision_analysis.py              # full run
python dnn_precision_analysis.py --smoke      # same pipeline, small subsets, few minutes
python dnn_precision_analysis.py --no-kaggle  # use the standard dataset mirrors
```

Useful options: `--epochs`, `--capture-batches`, `--reservoir`, `--out`, `--cpu`.
