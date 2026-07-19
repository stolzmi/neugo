package main

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"neugo/data"
	"neugo/nn"
	"neugo/train"
)

const (
	cifar10URL      = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
	cifar10Dir      = "dataset/cifar10"
	cifar10BatchOne = "dataset/cifar10/data_batch_1.bin"
	// maxRealSamples caps how many real images we train on in this demo.
	// The full batch (10000 images) trains correctly but takes a long time
	// with this library's pure-Go, non-SIMD conv/pooling loops — this cap
	// keeps `go run` fast while still exercising real data end to end.
	maxRealSamples = 500
)

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

func loadRealOrSynthetic() *data.CIFAR10Dataset {
	if _, err := os.Stat(cifar10BatchOne); err != nil {
		if downloadErr := downloadAndExtractTarGz(cifar10URL, cifar10Dir); downloadErr != nil {
			fmt.Println("could not download CIFAR-10, using synthetic data:", downloadErr)
			return syntheticCIFAR10(200)
		}
	}

	dataset, err := data.LoadCIFAR10Binary(cifar10BatchOne)
	if err != nil {
		fmt.Println("could not load downloaded CIFAR-10 file, using synthetic data:", err)
		return syntheticCIFAR10(200)
	}
	if len(dataset.Images) > maxRealSamples {
		dataset.Images = dataset.Images[:maxRealSamples]
		dataset.Labels = dataset.Labels[:maxRealSamples]
	}
	fmt.Printf("using %d real CIFAR-10 images\n", len(dataset.Images))
	return dataset
}

func main() {
	dataset := loadRealOrSynthetic()

	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{len(dataset.Images), 32, 32, 3},
		nn.Conv2D(rng, 3, 8, 3, nn.HeInit()),
		nn.ReLU(),
		nn.MaxPool2D(2, 2),
		nn.Conv2D(rng, 8, 16, 3, nn.HeInit()),
		nn.ReLU(),
		nn.MaxPool2D(2, 2),
		nn.Flatten(),
		nn.Linear(rng, 0, 64, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 64, 10, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}

	x := cifarImagesToTensor(dataset.Images)
	y := cifarLabelsToTensor(dataset.Labels)

	trainer := train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.CrossEntropy())
	if _, err := trainer.Fit(x, y, train.Epochs(15), train.BatchSize(16), train.Shuffle(true), train.Seed(2)); err != nil {
		fmt.Println("fit:", err)
		return
	}
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		fmt.Println("evaluate:", err)
		return
	}
	fmt.Printf("train-set accuracy: %.2f%% (classes: %v)\n", metrics.Accuracy, dataset.ClassNames)
}
