package data

import (
	"io"
	"math"
	"os"
)

type CIFAR10Dataset struct {
	Images     []*Image
	Labels     [][]float32
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

	images := make([]*Image, 0)
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

		img := NewImage(32, 32, 3)

		for c := 0; c < 3; c++ {
			for h := 0; h < 32; h++ {
				for w := 0; w < 32; w++ {
					idx := 1 + c*1024 + h*32 + w
					img.Data[h][w][c] = float32(buffer[idx]) / 255.0
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
	allImages := make([]*Image, 0)
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

	images := make([]*Image, 0)
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

		img := NewImage(32, 32, 3)

		for c := 0; c < 3; c++ {
			for h := 0; h < 32; h++ {
				for w := 0; w < 32; w++ {
					idx := 1 + c*1024 + h*32 + w
					img.Data[h][w][c] = float32(buffer[idx]) / 255.0
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

func NormalizeImages(images []*Image) []*Image {
	if len(images) == 0 {
		return images
	}
	numChannels, height, width := images[0].Channels, images[0].Height, images[0].Width
	means := make([]float32, numChannels)
	stds := make([]float32, numChannels)

	for c := 0; c < numChannels; c++ {
		var sum float32
		count := 0
		for _, img := range images {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					sum += img.Data[h][w][c]
					count++
				}
			}
		}
		means[c] = sum / float32(count)

		var variance float32
		for _, img := range images {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					diff := img.Data[h][w][c] - means[c]
					variance += diff * diff
				}
			}
		}
		stds[c] = float32(math.Sqrt(float64(variance / float32(count))))
	}

	normalized := make([]*Image, len(images))
	for i, img := range images {
		normalized[i] = NewImage(height, width, numChannels)
		for c := 0; c < numChannels; c++ {
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					if stds[c] > 0 {
						normalized[i].Data[h][w][c] = (img.Data[h][w][c] - means[c]) / stds[c]
					} else {
						normalized[i].Data[h][w][c] = img.Data[h][w][c] - means[c]
					}
				}
			}
		}
	}
	return normalized
}
