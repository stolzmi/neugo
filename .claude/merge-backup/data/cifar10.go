package data

import (
	"encoding/binary"
	"fmt"
	"io"
	"neugo/tensor"
	"os"
)

type CIFAR10Dataset struct {
	Images []*tensor.Tensor3D
	Labels [][]float32
	ClassNames []string
}

func LoadCIFAR10Binary(filepath string) (*CIFAR10Dataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	classNames := []string{
		"airplane", "automobile", "bird", "cat", "deer",
		"dog", "frog", "horse", "ship", "truck",
	}

	images := make([]*tensor.Tensor3D, 0)
	labels := make([][]float32, 0)

	buffer := make([]byte, 3073)

	for {
		n, err := io.ReadFull(file, buffer)
		if err == io.EOF {
			break
		}
		if err != nil || n != 3073 {
			break
		}

		label := int(buffer[0])
		oneHot := make([]float32, 10)
		oneHot[label] = 1.0
		labels = append(labels, oneHot)

		img := tensor.NewTensor3D(32, 32, 3)

		for c := 0; c < 3; c++ {
			for h := 0; h < 32; h++ {
				for w := 0; w < 32; w++ {
					idx := 1 + c*1024 + h*32 + w
					img.Data[c][h][w] = float32(buffer[idx]) / 255.0
				}
			}
		}

		images = append(images, img)
	}

	return &CIFAR10Dataset{
		Images:     images,
		Labels:     labels,
		ClassNames: classNames,
	}, nil
}

func LoadCIFAR10BinaryBatch(filepaths []string) (*CIFAR10Dataset, error) {
	allImages := make([]*tensor.Tensor3D, 0)
	allLabels := make([][]float32, 0)

	classNames := []string{
		"airplane", "automobile", "bird", "cat", "deer",
		"dog", "frog", "horse", "ship", "truck",
	}

	for _, filepath := range filepaths {
		dataset, err := LoadCIFAR10Binary(filepath)
		if err != nil {
			return nil, err
		}
		allImages = append(allImages, dataset.Images...)
		allLabels = append(allLabels, dataset.Labels...)
	}

	return &CIFAR10Dataset{
		Images:     allImages,
		Labels:     allLabels,
		ClassNames: classNames,
	}, nil
}

func LoadCIFAR10BinaryClassSubset(filepath string, targetClasses []int) (*CIFAR10Dataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	classNames := []string{
		"airplane", "automobile", "bird", "cat", "deer",
		"dog", "frog", "horse", "ship", "truck",
	}

	targetMap := make(map[int]bool)
	for _, c := range targetClasses {
		targetMap[c] = true
	}

	images := make([]*tensor.Tensor3D, 0)
	labels := make([][]float32, 0)

	buffer := make([]byte, 3073)

	for {
		n, err := io.ReadFull(file, buffer)
		if err == io.EOF {
			break
		}
		if err != nil || n != 3073 {
			break
		}

		label := int(buffer[0])

		if !targetMap[label] {
			continue
		}

		binaryLabel := float32(0)
		if label == targetClasses[1] {
			binaryLabel = 1
		}
		labels = append(labels, []float32{binaryLabel})

		img := tensor.NewTensor3D(32, 32, 3)

		for c := 0; c < 3; c++ {
			for h := 0; h < 32; h++ {
				for w := 0; w < 32; w++ {
					idx := 1 + c*1024 + h*32 + w
					img.Data[c][h][w] = float32(buffer[idx]) / 255.0
				}
			}
		}

		images = append(images, img)
	}

	subsetClassNames := []string{classNames[targetClasses[0]], classNames[targetClasses[1]]}

	return &CIFAR10Dataset{
		Images:     images,
		Labels:     labels,
		ClassNames: subsetClassNames,
	}, nil
}

func NormalizeImages(images []*tensor.Tensor3D) []*tensor.Tensor3D {
	if len(images) == 0 {
		return images
	}

	numChannels := images[0].Channels
	height := images[0].Height
	width := images[0].Width

	means := make([]float32, numChannels)
	stds := make([]float32, numChannels)

	for c := 0; c < numChannels; c++ {
		sum := float32(0.0)
		count := 0

		for _, img := range images {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					sum += img.Data[c][h][w]
					count++
				}
			}
		}

		means[c] = sum / float32(count)

		variance := float32(0.0)
		for _, img := range images {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					diff := img.Data[c][h][w] - means[c]
					variance += diff * diff
				}
			}
		}
		stds[c] = float32(sqrt(float64(variance / float32(count))))
	}

	normalized := make([]*tensor.Tensor3D, len(images))
	for i, img := range images {
		normalized[i] = tensor.NewTensor3D(height, width, numChannels)
		for c := 0; c < numChannels; c++ {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					if stds[c] > 0 {
						normalized[i].Data[c][h][w] = (img.Data[c][h][w] - means[c]) / stds[c]
					} else {
						normalized[i].Data[c][h][w] = img.Data[c][h][w] - means[c]
					}
				}
			}
		}
	}

	return normalized
}

func sqrt(x float64) float64 {
	if x == 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
