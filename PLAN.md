# Project Plan: Recognising Radio Signal Types with Deep Learning

**Automatic Modulation Classification using a Convolutional Neural Network (CNN)**

Group 18 · Department of ECE, NIT Agartala · Pre-Final Year Project (AY 2025–26)

**Supervisor:** Dr. Ashish Kumar Padhan

**Members:** Ankan Laskar (23UEC067) · Raj Kumar Gupta (23UEC085) · Beryl Debbarma (23UEC020) · Ashish Vaibhav Seema (23UEC091)

**Time available:** 2 weeks

---

## 1. The project in one paragraph

When a radio receiver picks up an unknown signal, the first question is: *how was this signal
modulated?* Is it BPSK, QPSK, QAM16, FM or something else? In a normal phone or Wi-Fi link the
receiver is told this in advance. In spectrum monitoring, cognitive radio and security work, nobody
tells it. Our project builds a program that looks at a very short piece of a received signal
(128 samples) and says which of 11 modulation types it is, **without any information about the
transmitter**. We use a Convolutional Neural Network (CNN), which learns to recognise the signal
types by itself from many labelled examples. We then compare it fairly against the traditional
method.

> **A simple way to picture it:** it is like recognising which language someone is speaking on a
> noisy phone line, without being told in advance.

## 2. Why this problem matters, and why the old methods fall short

There are two traditional ways to identify the modulation of a signal:

| Traditional method | How it works | The problem |
|---|---|---|
| **Likelihood-based** | Computes, for every possible modulation, how likely the received signal is, then picks the best one | Theoretically the best, but it needs exact knowledge of the channel (gain, noise, phase), and the computation is too heavy for real-time use |
| **Feature-based** | Engineers hand-pick statistics of the signal (for example higher-order cumulants, amplitude and phase variation), then a simple classifier decides | Works on clean signals but breaks down in noise. Every new modulation needs new features designed by hand |

**What a CNN changes:** it learns the useful patterns directly from the raw samples. Nobody has to
design features, it handles noise better, and adding a new modulation only needs new training data.

## 3. What we will prove

We will support three claims with measurements, not opinion:

1. **Better in noise.** The CNN is more accurate than the traditional method, especially when the
   signal is weak compared with the noise (low SNR).
2. **No hand-made features.** The CNN works on raw I/Q samples only.
3. **Fast enough for real use.** One decision takes only a few milliseconds on an ordinary laptop.

## 4. The data we use

We use **RadioML 2016.10a**, a public benchmark dataset made by DeepSig. Most research papers on
this topic use it, which lets us compare our results with published work.

| | |
|---|---|
| **Size** | 220,000 short signal clips |
| **Signal types (11)** | 8 digital: BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, CPFSK, GFSK. 3 analog: AM-DSB, AM-SSB, WBFM |
| **One clip** | 128 time samples, each with an I (in-phase) and a Q (quadrature) value |
| **Noise levels** | SNR from −20 dB (noise much stronger than the signal) to +18 dB (clean), in 2 dB steps |
| **Real-world effects included** | Fading (multipath), frequency offset, timing offset, noise |

**Fair testing.** We divide the clips into three groups:

- **60 % for learning**: the model trains on these.
- **20 % for checking progress**: used to decide when to stop training.
- **20 % kept aside as the final exam**: the model never sees these clips during training.

Every method (old and new) takes exactly the same final exam, so the comparison is fair.

## 5. How the system works

```
  Received signal  ─►  Scale to equal  ─►   Neural   ─►  A score for each  ─►  Answer:
 (128 I/Q samples)         power             network       of the 11 types      "QPSK (94 %)"
```

**What the CNN does, in simple words:** it slides small pattern detectors (called *filters*) along
the signal. The first layers notice simple things, such as a sudden phase jump or a change in
amplitude. Deeper layers combine these into bigger patterns, such as the shape of the constellation
or how the frequency keeps changing. During training the network adjusts its filters again and again
until it gives the right answers. Nobody tells it what to look for.

## 6. The three methods we compare

| # | Method | In simple words | Why it is in the project |
|---|---|---|---|
| 1 | **Traditional (feature-based)** | We calculate 24 well-known signal statistics (higher-order cumulants, amplitude, phase and frequency variations). A Decision Tree and a Random Forest make the decision | Represents "the old way" criticised in our proposal (Section 2.2) |
| 2 | **VT-CNN2** | The first well-known CNN for this problem, published by O'Shea et al. in 2016 | A published reference point, so our results can be compared with the literature |
| 3 | **Our proposed CNN (ResCNN-1D)** | A deeper but much smaller CNN. It uses *shortcut connections* (residual learning), which make deep networks easier to train. It has about 22 times fewer parameters than VT-CNN2, so it is lighter and faster | **Our main contribution** |

## 7. How we will measure success

We will produce three kinds of result:

- **Accuracy vs SNR graph** (the main result). One line per method, showing how accuracy changes from
  very noisy to clean signals.
- **Confusion matrix.** A table showing which signal types get mixed up with which.
- **Speed and size.** Time per decision in milliseconds, and the model's size.

**Our targets:**

| What we check | Target |
|---|---|
| Our CNN vs the traditional method | More accurate at every SNR from −10 dB to +18 dB |
| Our CNN vs VT-CNN2 | At least as accurate, with at least 10 times fewer parameters |
| Accuracy for reasonably clean signals (SNR ≥ 0 dB) | 85 % or more (published results are 85–90 %) |
| Speed | Less than 5 ms per decision on a laptop CPU |

**An honest expectation:** no method reaches 100 % on this dataset, because some pairs are genuinely
hard to tell apart:

- **QAM16 vs QAM64.** Their constellation points are packed closely together.
- **AM-DSB vs WBFM.** The dataset's analog signals carry recorded speech. During pauses in the speech,
  both signals reduce to a plain carrier and look identical.

We will point these out and explain them in the results instead of hiding them.

## 8. Two-week timeline

The code is already written, so the two weeks go into **getting results, understanding them and
writing them up**. Report writing starts on day 1, in parallel with the experiments, not at the end.

| Days | Experiments | Writing (in parallel) | Finished when |
|---|---|---|---|
| **1‑2** | Install tools, get the dataset, plot example signals of all 11 types | Introduction and problem statement (reuse the proposal) | Example plots ready |
| **3** | Run the traditional method (features + Decision Tree / Random Forest) | Literature review | Its accuracy-vs-SNR graph is ready |
| **4‑6** | Train VT-CNN2 and our CNN (can run overnight on a laptop) | System model and dataset chapter | Both models trained |
| **7** | Final comparison graph, confusion matrices, speed test | Methodology chapter | All results ready |
| **8‑10** | Study the results; re-run anything that looks wrong | Results and discussion chapter | Full report draft |
| **11‑12** | Prepare the live demo | Slides, and conclusion with future work | Slides ready |
| **13** | Full rehearsal with Sir's feedback | Final proof-reading | Ready to submit |
| **14** | **Buffer day** for anything that slipped | | Submitted |

## 9. Who does what

| Member | Main responsibility | Report chapter |
|---|---|---|
| Ankan Laskar | Dataset setup and example plots; traditional method run | System model and dataset |
| Raj Kumar Gupta | Training both CNNs, final comparison | Methodology |
| Beryl Debbarma | Confusion-matrix analysis, explaining the mistakes | Introduction and literature review |
| Ashish Vaibhav Seema | Speed test, live demo, slides | Results and conclusion |

Everyone reviews the full report before submission.

## 10. What we left out to fit two weeks

These are good ideas, but not needed to prove our three claims. They go into the report as
**future work**:

- Testing on real radio hardware (SDR, over-the-air signals).
- Data augmentation experiments (randomly rotating signals during training).
- Comparing performance on our own generated data against RadioML (the "synthetic gap").
- Newer architectures (CNN + LSTM, transformers) and running on a phone or embedded board.

## 11. Tools

Python, TensorFlow/Keras (to build the CNN), NumPy (numbers), scikit-learn (traditional
classifiers), Matplotlib/Seaborn (graphs). Google Colab's free GPU is used only if the laptop turns
out to be too slow. This is the same tool set as Section 5 of our proposal.

## 12. Risks and how we handle them

| What could go wrong | How we handle it |
|---|---|
| The dataset download is delayed | We download it manually on day 1. Meanwhile, our own signal generator (same format) lets us test all the code, so no time is wasted |
| Training is slow without a graphics card (GPU) | Our CNN is small. Training runs overnight; Colab is the backup |
| The model memorises the training data (overfitting) | A separate checking set, early stopping and dropout |
| Some signal types will always be confused | We explain why using the confusion matrix. It is a property of the data, not a bug |
| Writing gets squeezed at the end | Writing starts on day 1, alongside the experiments; day 14 is a buffer |

## 13. What we will submit

1. **Working code**, with step-by-step instructions to repeat our results.
2. **Trained models** (both CNNs and the traditional classifier).
3. **Results**: comparison graph, confusion matrices, speed table and a short written summary.
4. **Project report**: Introduction, Literature review, System model, Method, Results, Conclusion.
5. **Presentation with a live demo**: the model classifies random signals in front of the panel.

## 14. Current status

- **Done: all experiments, on the real RadioML 2016.10a dataset** (a checksum-verified copy from the
  Zenodo research archive, because DeepSig's own download page was broken).
- **Our CNN:** 86.1 % accuracy for SNR ≥ 0 dB (target 85 %: met). It beats VT-CNN2 (81.4 %) at all
  15 noise levels from −10 to +18 dB with 22.6× fewer parameters (met). It takes 1.16 ms per
  decision on the laptop CPU (target 5 ms: met).
- **Against the traditional method:** up to 18 points better in noisy conditions, and better at 13
  of 15 noise levels. On the two cleanest levels the two are tied within measurement noise.
- **Next:** write the report and slides, rehearse the live demo.

<div class="pb"></div>

## Appendix A: Model details (for technical questions)

**VT-CNN2** (O'Shea et al., 2016): about 2.83 million parameters

| Layer | What it does | Parameters |
|---|---|---|
| Conv 256 filters, size 1×3 | Finds short patterns in I and Q separately | 1,024 |
| Conv 80 filters, size 2×3 | Combines I and Q patterns | 122,960 |
| Dense 256 | Combines everything into a decision | 2,703,616 |
| Output, 11 values | One score per signal type | 2,827 |

**Our ResCNN-1D**: about 125 thousand parameters

| Layer | What it does | Output size |
|---|---|---|
| Starting conv, 32 filters | First look at the I/Q signal | 128 × 32 |
| Residual block 1 (32) + halve | Simple patterns | 64 × 32 |
| Residual block 2 (48) + halve | Combined patterns | 32 × 48 |
| Residual block 3 (64) + halve | Higher-level patterns | 16 × 64 |
| Residual block 4 (96) + halve | Signal-type patterns | 8 × 96 |
| Global average + max pooling | Summarises the whole clip | 192 |
| Dense 128, then output 11 | Final decision | 11 |

**Training settings:** Adam optimiser, learning rate 0.001. Up to 60 rounds over the data, stopping
early when the checking set stops improving. Dropout to avoid overfitting. A fixed random seed so the
results can be repeated exactly.

## Appendix B: Glossary

| Term | Meaning |
|---|---|
| **Modulation** | How information is placed on a radio wave (by changing its phase, amplitude or frequency) |
| **I/Q samples** | The in-phase and quadrature parts of a signal; together they give its amplitude and phase |
| **SNR** | Signal-to-Noise Ratio: how strong the signal is compared with the noise (higher = cleaner) |
| **Constellation** | A plot of I against Q. Each modulation has its own shape (e.g. 4 dots for QPSK) |
| **CNN** | Convolutional Neural Network: a model that learns pattern detectors from examples |
| **Parameter** | A number inside the network that training adjusts; more parameters = bigger, slower model |
| **Epoch** | One full pass of training over all the training data |
| **Overfitting** | When a model memorises its training data and does poorly on new data |
| **Confusion matrix** | A table showing, for each true signal type, what the model predicted |
| **Residual connection** | A shortcut path that skips a few layers, making deep networks easier to train |
