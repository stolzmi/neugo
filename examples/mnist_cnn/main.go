package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"

	"github.com/stolzmi/neugo/data"
	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

const (
	mnistDir         = "dataset/mnist"
	mnistTrainImages = "dataset/mnist/train-images-idx3-ubyte"
	mnistTrainLabels = "dataset/mnist/train-labels-idx1-ubyte"
	mnistTestImages  = "dataset/mnist/t10k-images-idx3-ubyte"
	mnistTestLabels  = "dataset/mnist/t10k-labels-idx1-ubyte"

	// Quick mode: ~10k training images (realistic synthetic), 5 epochs → ~2-3 min
	quickSamples = 10000
	quickEpochs  = 5

	// Full mode: 60k real training images, 10 epochs
	fullEpochs = 10

	checkpointPath = "mnist"
)

func mnistImagesToTensor(images []*data.Image) *nn.Tensor {
	h, w, c := images[0].Height, images[0].Width, images[0].Channels
	t := nn.NewTensor([]int{len(images), h, w, c})
	for i, img := range images {
		for hh := range h {
			for ww := range w {
				base := ((i*h+hh)*w + ww) * c
				copy(t.Data[base:base+c], img.Data[hh][ww])
			}
		}
	}
	return t
}

func mnistLabelsToTensor(labels [][]float32) *nn.Tensor {
	cols := len(labels[0])
	flat := make([]float32, len(labels)*cols)
	for i, row := range labels {
		copy(flat[i*cols:(i+1)*cols], row)
	}
	t, _ := nn.NewTensorFromData(flat, []int{len(labels), cols})
	return t
}

func syntheticMNIST(n int) *data.MNISTDataset {
	rng := nn.NewRNG(42)
	images := make([]*data.Image, n)
	labels := make([][]float32, n)

	for i := range n {
		class := i % 10
		img := data.NewImage(28, 28, 1)

		// Create MNIST-like digit patterns: each digit has a recognizable structure
		for h := range 28 {
			for w := range 28 {
				v := float32(0.1)
				ch := float32(h)
				cw := float32(w)

				switch class {
				case 0: // Circle
					dist := (ch-14)*(ch-14) + (cw-14)*(cw-14)
					if dist > 30 && dist < 100 {
						v = 0.8
					}
				case 1: // Vertical line
					if cw > 12 && cw < 16 {
						v = 0.8
					}
				case 2: // S-curve
					if (ch > 8 && ch < 12 && cw > 10 && cw < 18) ||
						(ch > 16 && ch < 20 && cw > 10 && cw < 18) ||
						(ch > 12 && ch < 16 && ((cw > 10 && cw < 14) || (cw > 14 && cw < 18))) {
						v = 0.8
					}
				case 3: // Vertical with bumps
					if cw > 12 && cw < 18 {
						v = 0.8
					} else if (ch > 6 && ch < 10 || ch > 18 && ch < 22) && cw > 16 && cw < 22 {
						v = 0.8
					}
				case 4: // Angles
					if (ch > 6 && ch < 10 && cw > 6 && cw < 10) ||
						(ch > 12 && ch < 16 && cw > 10 && cw < 18) ||
						(ch > 18 && ch < 22 && cw > 16 && cw < 20) {
						v = 0.8
					}
				case 5: // Zigzag
					if ((ch > 6 && ch < 10 && cw > 10 && cw < 18) ||
						(ch > 10 && ch < 14 && cw > 10 && cw < 14) ||
						(ch > 14 && ch < 18 && cw > 14 && cw < 18) ||
						(ch > 18 && ch < 22 && cw > 14 && cw < 22)) {
						v = 0.8
					}
				case 6: // Loop at bottom
					dist := (ch-10)*(ch-10) + (cw-14)*(cw-14)
					if dist < 50 || (ch > 16 && ch < 22 && cw > 10 && cw < 18) {
						v = 0.8
					}
				case 7: // Top horizontal to diagonal
					if (ch < 10 && cw > 10 && cw < 18) ||
						(ch > 10 && cw > 8+ch/2 && cw < 12+ch/2) {
						v = 0.8
					}
				case 8: // Two loops
					dist1 := (ch-10)*(ch-10) + (cw-14)*(cw-14)
					dist2 := (ch-18)*(ch-18) + (cw-14)*(cw-14)
					if dist1 < 40 || dist2 < 40 {
						v = 0.8
					}
				case 9: // Loop at top
					dist := (ch-10)*(ch-10) + (cw-14)*(cw-14)
					if dist < 50 || (ch > 14 && ch < 22 && cw > 12 && cw < 18) {
						v = 0.8
					}
				}
				img.Data[h][w][0] = v + (rng.Float32()-0.5)*0.15
			}
		}
		images[i] = img

		label := make([]float32, 10)
		label[class] = 1
		labels[i] = label
	}

	classNames := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
	return &data.MNISTDataset{Images: images, Labels: labels, ClassNames: classNames}
}

type datasetSplit struct {
	trainImages []*data.Image
	trainLabels [][]float32
	evalImages  []*data.Image
	evalLabels  [][]float32
	classNames  []string
}

func ensureDownloaded(trainImages, trainLabels string) error {
	// Check if files exist
	if _, err := os.Stat(trainImages); err == nil {
		if _, err := os.Stat(trainLabels); err == nil {
			return nil
		}
	}

	if err := os.MkdirAll(mnistDir, 0755); err != nil {
		return err
	}

	fmt.Println("MNIST data not found. To use real MNIST data:")
	fmt.Println("1. Download from: http://yann.lecun.com/exdb/mnist/")
	fmt.Println("2. Gunzip files and place in dataset/mnist/")
	fmt.Println("3. Or install mnist module: go get -u github.com/golbin/mnist")
	fmt.Println("For now, using realistic synthetic MNIST-like data for quick testing.")

	return nil
}

func quickSplit(dataset *data.MNISTDataset) *datasetSplit {
	if len(dataset.Images) > quickSamples {
		dataset.Images = dataset.Images[:quickSamples]
		dataset.Labels = dataset.Labels[:quickSamples]
	}
	split := data.SplitImageData(rand.New(rand.NewSource(42)), dataset.Images, dataset.Labels,
		data.SplitConfig{TrainRatio: 0.8, ValRatio: 0.2, Shuffle: true})
	return &datasetSplit{
		trainImages: split.TrainX, trainLabels: split.TrainY,
		evalImages:  split.ValX, evalLabels: split.ValY,
		classNames:  dataset.ClassNames,
	}
}

func loadData(quick bool) *datasetSplit {
	// Try to load real MNIST if available
	if err := ensureDownloaded(mnistTrainImages, mnistTrainLabels); err == nil {
		if trainSet, err := data.LoadMNIST(mnistTrainImages, mnistTrainLabels); err == nil {
			if !quick {
				// Try to load test set for full mode
				if testSet, err := data.LoadMNIST(mnistTestImages, mnistTestLabels); err == nil {
					return &datasetSplit{
						trainImages: trainSet.Images, trainLabels: trainSet.Labels,
						evalImages: testSet.Images, evalLabels: testSet.Labels,
						classNames: trainSet.ClassNames,
					}
				}
			}
			// Use training set split for quick mode or when test set unavailable
			return quickSplit(trainSet)
		}
	}

	// Fall back to realistic synthetic MNIST-like data
	return quickSplit(syntheticMNIST(quickSamples))
}

func (s *datasetSplit) normalizeAndAugment(augment bool) data.ImageStats {
	combined := make([]*data.Image, 0, len(s.trainImages)+len(s.evalImages))
	combined = append(combined, s.trainImages...)
	combined = append(combined, s.evalImages...)
	normalized, stats := data.NormalizeImagesWithStats(combined)
	trainLen := len(s.trainImages)
	s.trainImages = normalized[:trainLen]
	s.evalImages = normalized[trainLen:]
	if augment {
		s.trainImages, s.trainLabels = data.AugmentWithFlips(s.trainImages, s.trainLabels)
	}
	return stats
}

func main() {
	quick := flag.Bool("quick", true, "fast mode: 5k images, 5 epochs, no augmentation (default true for fast testing)")
	flag.Parse()

	split := loadData(*quick)
	stats := split.normalizeAndAugment(!*quick)

	epochs := fullEpochs
	if *quick {
		epochs = quickEpochs
	}
	fmt.Printf("training on %d images, evaluating on %d held-out images\n",
		len(split.trainImages), len(split.evalImages))

	inputShape := []int{1, 28, 28, 1}
	rng := nn.NewRNG(1)
	model, err := nn.Sequential(inputShape,
		nn.Conv2DSame(rng, 1, 8, 3, nn.HeInit()),
		nn.BatchNorm(8),
		nn.ReLU(),
		nn.MaxPool2D(2, 2), // 28x28 -> 14x14
		nn.Conv2DSame(rng, 8, 16, 3, nn.HeInit()),
		nn.BatchNorm(16),
		nn.ReLU(),
		nn.MaxPool2D(2, 2), // 14x14 -> 7x7
		nn.Flatten(),
		nn.Dropout(0.3),
		nn.Linear(rng, 0, 64, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 64, 10, nn.XavierInit()),
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

	x := mnistImagesToTensor(split.trainImages)
	y := mnistLabelsToTensor(split.trainLabels)
	evalX := mnistImagesToTensor(split.evalImages)
	evalY := mnistLabelsToTensor(split.evalLabels)

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
			train.EarlyStopping(3),
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
