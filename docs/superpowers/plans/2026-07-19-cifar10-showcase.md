# CIFAR-10 Showcase Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `examples/cifar10_cnn` into a full-capability showcase: 50k-image training with flip augmentation, a BatchNorm/Dropout CNN, cosine LR annealing, early stopping, best-model checkpointing, and evaluation on the real CIFAR-10 test batch — with a `-quick` flag preserving the old fast behavior.

**Architecture:** Single `main` package example (`examples/cifar10_cnn/main.go`) plus a unit-test file for the augmentation helpers. No library (`nn/`, `train/`, `data/`) changes — the point is to exercise the existing API surface. Spec: `docs/superpowers/specs/2026-07-19-cifar10-showcase-design.md`.

**Tech Stack:** Go (pure stdlib + this repo's `neugo/nn`, `neugo/train`, `neugo/data` packages).

## Global Constraints

- No changes outside `examples/cifar10_cnn/`.
- No new dependencies; `go.mod` untouched.
- Keep the synthetic-data fallback when download fails.
- Keep the existing download/extract helper (`downloadAndExtractTarGz`) exactly as-is.
- All commands below run from repo root `C:\projects\neugo`.
- The dataset is already downloaded at `dataset/cifar10/` (all 5 train batches + `test_batch.bin`), so runs are offline.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Horizontal-flip augmentation helpers

**Files:**
- Modify: `examples/cifar10_cnn/main.go` (add two functions, no removals yet)
- Test: `examples/cifar10_cnn/main_test.go` (create)

**Interfaces:**
- Consumes: `data.Image` / `data.NewImage(height, width, channels int) *data.Image` from `neugo/data`.
- Produces: `flipHorizontal(img *data.Image) *data.Image` (returns a new mirrored image, input untouched) and `augmentWithFlips(images []*data.Image, labels [][]float32) ([]*data.Image, [][]float32)` (returns original+flipped pairs interleaved: `img0, flip0, img1, flip1, …`, labels duplicated pairwise). Task 2's `normalizeAndAugment` calls `augmentWithFlips`.

- [ ] **Step 1: Write the failing tests**

Create `examples/cifar10_cnn/main_test.go`:

```go
package main

import (
	"testing"

	"neugo/data"
)

func TestFlipHorizontalMirrorsWidthAndKeepsOriginal(t *testing.T) {
	img := data.NewImage(2, 3, 1)
	img.Data[0][0][0], img.Data[0][1][0], img.Data[0][2][0] = 1, 2, 3
	img.Data[1][0][0], img.Data[1][1][0], img.Data[1][2][0] = 4, 5, 6

	flipped := flipHorizontal(img)

	want := [][]float32{{3, 2, 1}, {6, 5, 4}}
	for h := 0; h < 2; h++ {
		for w := 0; w < 3; w++ {
			if got := flipped.Data[h][w][0]; got != want[h][w] {
				t.Errorf("flipped[%d][%d] = %v, want %v", h, w, got, want[h][w])
			}
		}
	}
	if img.Data[0][0][0] != 1 || img.Data[0][2][0] != 3 {
		t.Error("flipHorizontal mutated its input image")
	}
}

func TestAugmentWithFlipsDoublesAndPairsLabels(t *testing.T) {
	a := data.NewImage(1, 2, 1)
	a.Data[0][0][0], a.Data[0][1][0] = 1, 2
	b := data.NewImage(1, 2, 1)
	b.Data[0][0][0], b.Data[0][1][0] = 3, 4
	labels := [][]float32{{1, 0}, {0, 1}}

	outImgs, outLabels := augmentWithFlips([]*data.Image{a, b}, labels)

	if len(outImgs) != 4 || len(outLabels) != 4 {
		t.Fatalf("got %d images, %d labels, want 4 and 4", len(outImgs), len(outLabels))
	}
	// order: original, its flip, next original, its flip
	if outImgs[0] != a || outImgs[2] != b {
		t.Error("originals not at even indices")
	}
	if outImgs[1].Data[0][0][0] != 2 || outImgs[3].Data[0][0][0] != 4 {
		t.Error("flips not at odd indices or not mirrored")
	}
	for i, wantLabel := range [][]float32{{1, 0}, {1, 0}, {0, 1}, {0, 1}} {
		for j := range wantLabel {
			if outLabels[i][j] != wantLabel[j] {
				t.Errorf("label[%d] = %v, want %v", i, outLabels[i], wantLabel)
			}
		}
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./examples/cifar10_cnn/ -run TestFlip -v` (and `-run TestAugment`)
Expected: FAIL to build with `undefined: flipHorizontal` / `undefined: augmentWithFlips`.

- [ ] **Step 3: Implement the helpers**

Add to `examples/cifar10_cnn/main.go` (below `cifarLabelsToTensor`):

```go
// flipHorizontal returns a new image mirrored left-to-right. The library
// has no built-in augmentation, so the example implements the classic
// CIFAR-10 horizontal flip itself.
func flipHorizontal(img *data.Image) *data.Image {
	out := data.NewImage(img.Height, img.Width, img.Channels)
	for h := 0; h < img.Height; h++ {
		for w := 0; w < img.Width; w++ {
			copy(out.Data[h][w], img.Data[h][img.Width-1-w])
		}
	}
	return out
}

// augmentWithFlips doubles the dataset by interleaving each image with its
// horizontal mirror, duplicating labels pairwise.
func augmentWithFlips(images []*data.Image, labels [][]float32) ([]*data.Image, [][]float32) {
	outImages := make([]*data.Image, 0, len(images)*2)
	outLabels := make([][]float32, 0, len(labels)*2)
	for i, img := range images {
		outImages = append(outImages, img, flipHorizontal(img))
		outLabels = append(outLabels, labels[i], labels[i])
	}
	return outImages, outLabels
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./examples/cifar10_cnn/ -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/cifar10_cnn/main.go examples/cifar10_cnn/main_test.go
git commit -m "feat(examples): add horizontal-flip augmentation helpers to cifar10_cnn"
```

---

### Task 2: Rewrite the example — data pipeline, showcase model, training loop

**Files:**
- Modify: `examples/cifar10_cnn/main.go` (full rewrite except `downloadAndExtractTarGz`, `syntheticCIFAR10`, `cifarImagesToTensor`, `cifarLabelsToTensor`, and Task 1's helpers, which stay verbatim)

**Interfaces:**
- Consumes: Task 1's `augmentWithFlips`; library API: `data.LoadCIFAR10Binary`, `data.LoadCIFAR10BinaryBatch`, `data.NormalizeImages`, `data.SplitImageData`, `nn.Sequential`, `nn.Conv2DSame`, `nn.BatchNorm`, `nn.Dropout`, `nn.GELU`, `nn.Summary`, `nn.Save`, `train.New`, `train.Adam`, `train.CrossEntropy`, `train.CosineAnnealing`, `train.EarlyStopping`, `train.ModelCheckpoint`, `train.ProgressBar`, `train.ClipGrad`, `train.WithSaveFunc`, `train.FormatConfusionMatrix`.
- Produces: the final example binary; `datasetSplit` struct internal to the example.

- [ ] **Step 1: Replace constants and add the train-batch list**

In `examples/cifar10_cnn/main.go`, replace the existing `const` block (which defines `cifar10URL`, `cifar10Dir`, `cifar10BatchOne`, `maxRealSamples`, `epochs`) with:

```go
const (
	cifar10URL     = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
	cifar10Dir     = "dataset/cifar10"
	cifar10TestBin = "dataset/cifar10/test_batch.bin"

	// quick mode preserves the old fast demo: one batch file capped at
	// quickSamples images, 80/20 split, no augmentation — minutes, not
	// hours, despite the library's pure-Go, non-SIMD conv loops.
	quickSamples = 5000
	quickEpochs  = 15

	// full mode is the showcase: all 50k train images (flip-augmented to
	// 100k) against the official 10k test batch. Expect an overnight run.
	fullEpochs = 40

	checkpointPath = "cifar_10"
)

var cifar10TrainBins = []string{
	"dataset/cifar10/data_batch_1.bin",
	"dataset/cifar10/data_batch_2.bin",
	"dataset/cifar10/data_batch_3.bin",
	"dataset/cifar10/data_batch_4.bin",
	"dataset/cifar10/data_batch_5.bin",
}
```

- [ ] **Step 2: Replace `loadRealOrSynthetic` with the mode-aware pipeline**

Delete `loadRealOrSynthetic` and add:

```go
// datasetSplit is train + held-out eval data ready for tensor conversion.
type datasetSplit struct {
	trainImages []*data.Image
	trainLabels [][]float32
	evalImages  []*data.Image
	evalLabels  [][]float32
	classNames  []string
}

func ensureDownloaded() error {
	if _, err := os.Stat(cifar10TrainBins[0]); err == nil {
		return nil
	}
	return downloadAndExtractTarGz(cifar10URL, cifar10Dir)
}

// quickSplit caps the dataset at quickSamples images and carves out a 20%
// validation split — the old demo behavior, also the fallback when the
// full-mode test batch is unavailable.
func quickSplit(dataset *data.CIFAR10Dataset) *datasetSplit {
	if len(dataset.Images) > quickSamples {
		dataset.Images = dataset.Images[:quickSamples]
		dataset.Labels = dataset.Labels[:quickSamples]
	}
	split := data.SplitImageData(rand.New(rand.NewSource(42)), dataset.Images, dataset.Labels,
		data.SplitConfig{TrainRatio: 0.8, ValRatio: 0.2, Shuffle: true})
	return &datasetSplit{
		trainImages: split.TrainX, trainLabels: split.TrainY,
		evalImages: split.ValX, evalLabels: split.ValY,
		classNames: dataset.ClassNames,
	}
}

func loadShowcaseData(quick bool) *datasetSplit {
	if err := ensureDownloaded(); err != nil {
		fmt.Println("could not download CIFAR-10, using synthetic data:", err)
		return quickSplit(syntheticCIFAR10(200))
	}
	if quick {
		dataset, err := data.LoadCIFAR10Binary(cifar10TrainBins[0])
		if err != nil {
			fmt.Println("could not load CIFAR-10 batch, using synthetic data:", err)
			return quickSplit(syntheticCIFAR10(200))
		}
		return quickSplit(dataset)
	}
	trainSet, err := data.LoadCIFAR10BinaryBatch(cifar10TrainBins)
	if err != nil {
		fmt.Println("could not load CIFAR-10 batches, using synthetic data:", err)
		return quickSplit(syntheticCIFAR10(200))
	}
	testSet, err := data.LoadCIFAR10Binary(cifar10TestBin)
	if err != nil {
		fmt.Println("could not load test batch, splitting training data instead:", err)
		return quickSplit(trainSet)
	}
	return &datasetSplit{
		trainImages: trainSet.Images, trainLabels: trainSet.Labels,
		evalImages: testSet.Images, evalLabels: testSet.Labels,
		classNames: trainSet.ClassNames,
	}
}

// normalizeAndAugment standardizes every channel to zero mean / unit std
// (stats over train+eval together — trivial leakage, acceptable for a
// demo) and, when augment is set, doubles the training set with mirrors.
func (s *datasetSplit) normalizeAndAugment(augment bool) {
	combined := make([]*data.Image, 0, len(s.trainImages)+len(s.evalImages))
	combined = append(combined, s.trainImages...)
	combined = append(combined, s.evalImages...)
	normalized := data.NormalizeImages(combined)
	s.trainImages = normalized[:len(s.trainImages)]
	s.evalImages = normalized[len(s.trainImages):]
	if augment {
		s.trainImages, s.trainLabels = augmentWithFlips(s.trainImages, s.trainLabels)
	}
}
```

- [ ] **Step 3: Rewrite `main` with the showcase model and training setup**

Replace the entire `main` function with:

```go
func main() {
	quick := flag.Bool("quick", false, "fast smoke-test mode: one batch, 5k images, 15 epochs, no augmentation")
	flag.Parse()

	split := loadShowcaseData(*quick)
	split.normalizeAndAugment(!*quick)

	epochs := fullEpochs
	if *quick {
		epochs = quickEpochs
	}
	fmt.Printf("training on %d images, evaluating on %d held-out images\n",
		len(split.trainImages), len(split.evalImages))

	inputShape := []int{1, 32, 32, 3}
	rng := nn.NewRNG(1)
	model, err := nn.Sequential(inputShape,
		nn.Conv2DSame(rng, 3, 16, 3, nn.HeInit()),
		nn.BatchNorm(16),
		nn.ReLU(),
		nn.MaxPool2D(2, 2), // 32x32 -> 16x16
		nn.Conv2DSame(rng, 16, 32, 3, nn.HeInit()),
		nn.BatchNorm(32),
		nn.ReLU(),
		nn.MaxPool2D(2, 2), // 16x16 -> 8x8
		nn.Conv2DSame(rng, 32, 64, 3, nn.HeInit()),
		nn.BatchNorm(64),
		nn.ReLU(),
		nn.MaxPool2D(2, 2), // 8x8 -> 4x4
		nn.Flatten(),
		nn.Dropout(0.5),
		nn.Linear(rng, 0, 128, nn.HeInit()),
		nn.GELU(),
		nn.Linear(rng, 128, 10, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}

	summary, err := nn.Summary(model, inputShape)
	if err != nil {
		fmt.Println("summary:", err)
		return
	}
	fmt.Println(summary)

	x := cifarImagesToTensor(split.trainImages)
	y := cifarLabelsToTensor(split.trainLabels)
	evalX := cifarImagesToTensor(split.evalImages)
	evalY := cifarLabelsToTensor(split.evalLabels)

	opt := train.Adam(1e-3, 0.9, 0.999, 1e-8)
	trainer := train.New(model, opt, train.CrossEntropy())
	hist, err := trainer.Fit(x, y,
		train.Epochs(epochs), train.BatchSize(32), train.Shuffle(true), train.Seed(2),
		train.ClipGrad(5),
		train.Validation(evalX, evalY),
		train.WithSaveFunc(nn.Save),
		train.Callbacks(
			train.ProgressBar(epochs, 1),
			train.CosineAnnealing(opt, 1e-5, epochs),
			train.EarlyStopping(6),
			train.ModelCheckpoint(checkpointPath, "loss", "min", true),
		),
	)
	if err != nil {
		fmt.Println("fit:", err)
		return
	}

	fmt.Println(hist.PlotLoss(60, 12))

	metrics, err := trainer.Evaluate(evalX, evalY)
	if err != nil {
		fmt.Println("evaluate:", err)
		return
	}
	fmt.Printf("held-out evaluation: loss %.4f - acc %.2f%% - precision %.2f - recall %.2f - f1 %.2f\n",
		metrics.Loss, metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1Score)
	fmt.Println(train.FormatConfusionMatrix(&metrics, split.classNames))
	fmt.Printf("best model (lowest val loss) checkpointed to %s\n", checkpointPath)
}
```

Notes for the implementer:
- Add `"flag"` to the imports; all other imports stay (`os`, `math/rand`, `time`, etc. are still used by the kept helpers).
- The trailing `nn.Save(model, "cifar_10")` call from the old `main` is gone on purpose: `ModelCheckpoint` + `WithSaveFunc(nn.Save)` now saves the *best* epoch during training, and `EarlyStopping(6)` restores best weights so the final `Evaluate` matches the checkpoint.
- `Softmax` must remain the last module so `train.New` keeps the fused softmax+cross-entropy backward path.
- `train.CosineAnnealing(opt, 1e-5, epochs)` must be constructed *after* `opt` so it captures the initial LR 1e-3.

- [ ] **Step 4: Build, vet, test**

Run: `go build ./... && go vet ./... && go test ./examples/cifar10_cnn/`
Expected: all succeed; Task 1 tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/cifar10_cnn/main.go
git commit -m "feat(examples): turn cifar10_cnn into a full library showcase"
```

---

### Task 3: End-to-end verification (quick mode)

**Files:** none modified (fixes only if the run reveals bugs).

**Interfaces:** consumes the finished example from Task 2.

- [ ] **Step 1: Record the checkpoint file's pre-run timestamp**

Run: `Get-Item cifar_10 | Select-Object LastWriteTime` (PowerShell) — the file exists from the old run.

- [ ] **Step 2: Run quick mode end-to-end**

Run: `go run ./examples/cifar10_cnn -quick` (background; expect roughly 15–60 minutes — the model is ~7x the FLOPs of the old one).
Expected output, in order:
1. `training on 4000 images, evaluating on 1000 held-out images`
2. an `nn.Summary` table listing Conv2D/BatchNorm/ReLU/MaxPool2D/Flatten/Dropout/Linear/GELU/Softmax rows and a total parameter count (~190k)
3. `Epoch 1/15 … Epoch N/15` lines with val loss/acc (N ≤ 15; early stopping may end sooner)
4. the ASCII loss plot
5. `held-out evaluation: …` metrics line — val accuracy should beat the old 48.7% final (expect mid-50s%+, and crucially val loss should NOT diverge from train loss the way the old run did)
6. the confusion matrix
7. `best model (lowest val loss) checkpointed to cifar_10`

- [ ] **Step 3: Verify the checkpoint was rewritten during training**

Run: `Get-Item cifar_10 | Select-Object LastWriteTime`
Expected: newer than Step 1's timestamp.

- [ ] **Step 4: Full test suite**

Run: `go test ./...`
Expected: all packages PASS.

- [ ] **Step 5: Commit any fixes; otherwise nothing to commit**

If the run surfaced fixes, commit them with a `fix(examples): …` message.

---

## Self-Review (completed)

- **Spec coverage:** modes/flag → Task 2 Steps 1+3; data pipeline incl. batch loading, test batch, normalization → Task 2 Step 2; augmentation → Task 1 + `normalizeAndAugment`; model → Task 2 Step 3; training config & callbacks → Task 2 Step 3; reporting → Task 2 Step 3; error-handling fallbacks → `loadShowcaseData`; verification → Task 3. No gaps.
- **Placeholder scan:** none.
- **Type consistency:** `datasetSplit` fields, `augmentWithFlips` signature, and constant names match across tasks.
