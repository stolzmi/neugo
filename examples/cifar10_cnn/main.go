package main

import (
	"archive/tar"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/stolzmi/neugo/data"
	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

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

func cifarImagesToTensor(images []*data.Image) *nn.Tensor {
	h, w, c := images[0].Height, images[0].Width, images[0].Channels
	t := nn.NewTensor([]int{len(images), h, w, c})
	for i, img := range images {
		for hh := 0; hh < h; hh++ {
			for ww := 0; ww < w; ww++ {
				base := ((i*h+hh)*w + ww) * c
				copy(t.Data[base:base+c], img.Data[hh][ww])
			}
		}
	}
	return t
}

func cifarLabelsToTensor(labels [][]float32) *nn.Tensor {
	cols := len(labels[0])
	flat := make([]float32, len(labels)*cols)
	for i, row := range labels {
		copy(flat[i*cols:(i+1)*cols], row)
	}
	t, _ := nn.NewTensorFromData(flat, []int{len(labels), cols})
	return t
}

func syntheticCIFAR10(n int) *data.CIFAR10Dataset {
	rng := nn.NewRNG(21)
	classNames := []string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}
	images := make([]*data.Image, n)
	labels := make([][]float32, n)
	for i := 0; i < n; i++ {
		class := i % 10
		img := data.NewImage(32, 32, 3)
		for h := 0; h < 32; h++ {
			for w := 0; w < 32; w++ {
				for c := 0; c < 3; c++ {
					v := float32(0.1)
					if (h+w+c)%10 == class {
						v = 0.9
					}
					img.Data[h][w][c] = v + (rng.Float32()-0.5)*0.05
				}
			}
		}
		images[i] = img
		label := make([]float32, 10)
		label[class] = 1
		labels[i] = label
	}
	return &data.CIFAR10Dataset{Images: images, Labels: labels, ClassNames: classNames}
}

// downloadAndExtractTarGz downloads a .tar.gz archive from url and
// extracts every regular file it contains directly into destDir (flattening
// any subdirectories in the archive, e.g. cifar-10-batches-bin/*.bin lands
// straight in destDir/*.bin). Flattening via filepath.Base also means a
// malicious or corrupt archive entry can never write outside destDir
// ("zip-slip") — only the final path element of each entry name is ever
// used, so ".." components in an entry name are stripped, not honored.
func downloadAndExtractTarGz(url, destDir string) error {
	client := &http.Client{Timeout: 30 * time.Minute}
	fmt.Printf("downloading %s (this may take a while) ...\n", url)
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download: unexpected status %s", resp.Status)
	}

	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("gzip: %w", err)
	}
	defer gz.Close()

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("mkdir %s: %w", destDir, err)
	}

	tr := tar.NewReader(gz)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar: %w", err)
		}
		if header.Typeflag != tar.TypeReg {
			continue
		}
		name := filepath.Base(header.Name)
		if name == "" || name == "." || name == ".." {
			continue
		}
		target := filepath.Join(destDir, name)

		out, err := os.Create(target)
		if err != nil {
			return fmt.Errorf("create %s: %w", target, err)
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			return fmt.Errorf("write %s: %w", target, err)
		}
		out.Close()
	}
	fmt.Printf("extracted CIFAR-10 to %s\n", destDir)
	return nil
}

// datasetSplit is train + held-out eval data ready for tensor conversion.
type datasetSplit struct {
	trainImages []*data.Image
	trainLabels [][]float32
	evalImages  []*data.Image
	evalLabels  [][]float32
	classNames  []string
}

// ensureDownloaded fetches the archive unless every file in required is
// already present. Quick mode only needs the first batch file, so a
// partially-extracted dataset (e.g. just data_batch_1.bin from an older
// version of this example) must not be mistaken for a complete one.
func ensureDownloaded(required []string) error {
	missing := false
	for _, path := range required {
		if _, err := os.Stat(path); err != nil {
			missing = true
			break
		}
	}
	if !missing {
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
	required := append(append([]string{}, cifar10TrainBins...), cifar10TestBin)
	if quick {
		required = cifar10TrainBins[:1]
	}
	if err := ensureDownloaded(required); err != nil {
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
// It returns the normalization stats so main can bundle them into the
// saved model via nn.SaveWithMetadata — without them, running inference
// on a fresh image later would require separately remembering exactly
// how this run normalized its training data.
func (s *datasetSplit) normalizeAndAugment(augment bool) data.ImageStats {
	combined := make([]*data.Image, 0, len(s.trainImages)+len(s.evalImages))
	combined = append(combined, s.trainImages...)
	combined = append(combined, s.evalImages...)
	normalized, stats := data.NormalizeImagesWithStats(combined)
	s.trainImages = normalized[:len(s.trainImages)]
	s.evalImages = normalized[len(s.trainImages):]
	if augment {
		s.trainImages, s.trainLabels = data.AugmentWithFlips(s.trainImages, s.trainLabels)
	}
	return stats
}

func main() {
	quick := flag.Bool("quick", false, "fast smoke-test mode: one batch, 5k images, 15 epochs, no augmentation")
	flag.Parse()

	split := loadShowcaseData(*quick)
	stats := split.normalizeAndAugment(!*quick)

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

	meta := nn.Metadata{
		InputShape:    inputShape,
		ClassNames:    split.classNames,
		Normalization: &nn.NormalizationStats{Mean: stats.Mean, Std: stats.Std},
	}
	saveCheckpoint := func(m *nn.SequentialModel, path string) error {
		return nn.SaveWithMetadata(m, path, meta)
	}

	opt := train.Adam(1e-3, 0.9, 0.999, 1e-8)
	trainer := train.New(model, opt, train.CrossEntropy())
	hist, err := trainer.Fit(x, y,
		train.Epochs(epochs), train.BatchSize(32), train.Shuffle(true), train.Seed(2),
		train.ClipGrad(5),
		train.Validation(evalX, evalY),
		train.WithSaveFunc(saveCheckpoint),
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
