package main

import (
	"fmt"
	"github.com/stolzmi/neugo/data"
	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

func imagesToTensor(images []*data.Image) *nn.Tensor {
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

func labelsToTensor(labels [][]float32) *nn.Tensor {
	cols := len(labels[0])
	flat := make([]float32, len(labels)*cols)
	for i, row := range labels {
		copy(flat[i*cols:(i+1)*cols], row)
	}
	t, _ := nn.NewTensorFromData(flat, []int{len(labels), cols})
	return t
}

// syntheticFashionMNIST stands in for real data when no CSV is present —
// see the note at the top of Task 23.
func syntheticFashionMNIST(n int) *data.ImageDataset {
	rng := nn.NewRNG(11)
	images := make([]*data.Image, n)
	labels := make([][]float32, n)
	for i := 0; i < n; i++ {
		class := i % 10
		img := data.NewImage(28, 28, 1)
		for h := 0; h < 28; h++ {
			for w := 0; w < 28; w++ {
				v := float32(0.1)
				if (h+w)%10 == class {
					v = 0.9
				}
				img.Data[h][w][0] = v + (rng.Float32()-0.5)*0.05
			}
		}
		images[i] = img
		label := make([]float32, 10)
		label[class] = 1
		labels[i] = label
	}
	return &data.ImageDataset{Images: images, Labels: labels, Height: 28, Width: 28, Channels: 1}
}

func main() {
	dataset, err := data.LoadMNISTFromCSV("dataset/fashion_mnist/fashion-mnist_train.csv")
	if err != nil {
		fmt.Println("no Fashion-MNIST CSV found, using synthetic data:", err)
		dataset = syntheticFashionMNIST(200)
	}

	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{len(dataset.Images), dataset.Height, dataset.Width, dataset.Channels},
		nn.Conv2DSame(rng, dataset.Channels, 8, 3, nn.HeInit()),
		nn.ReLU(),
		nn.BatchNorm(8),
		nn.MaxPool2D(2, 2),
		nn.Conv2DSame(rng, 8, 16, 3, nn.HeInit()),
		nn.ReLU(),
		nn.MaxPool2D(2, 2),
		nn.Flatten(),
		nn.Linear(rng, 0, 64, nn.HeInit()),
		nn.ReLU(),
		nn.Dropout(0.3),
		nn.Linear(rng, 64, 10, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}

	x := imagesToTensor(dataset.Images)
	y := labelsToTensor(dataset.Labels)

	trainer := train.New(model, train.Adam(1e-3, 0.9, 0.999, 1e-8), train.CrossEntropy())
	if _, err := trainer.Fit(x, y, train.Epochs(20), train.BatchSize(16), train.Shuffle(true), train.Seed(2)); err != nil {
		fmt.Println("fit:", err)
		return
	}
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		fmt.Println("evaluate:", err)
		return
	}
	fmt.Printf("train-set accuracy: %.2f%%\n", metrics.Accuracy)
}
