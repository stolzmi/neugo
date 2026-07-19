package data

import (
	"encoding/csv"
	"fmt"
	"neugo/tensor"
	"os"
	"strconv"
)

type ImageDataset struct {
	Images []*tensor.Tensor3D
	Labels [][]float32
	Height int
	Width  int
	Channels int
}

func LoadMNISTFromCSV(filepath string) (*ImageDataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	hasHeader := false
	startIdx := 0
	if _, err := strconv.ParseFloat(records[0][0], 32); err != nil {
		hasHeader = true
		startIdx = 1
	}

	numSamples := len(records) - startIdx
	images := make([]*tensor.Tensor3D, numSamples)
	labels := make([][]float32, numSamples)

	height := 28
	width := 28
	channels := 1

	for i := startIdx; i < len(records); i++ {
		record := records[i]

		label, err := strconv.ParseFloat(record[0], 32)
		if err != nil {
			return nil, fmt.Errorf("invalid label at row %d: %v", i, err)
		}

		labels[i-startIdx] = make([]float32, 10)
		labels[i-startIdx][int(label)] = 1.0

		image := tensor.NewTensor3D(height, width, channels)

		for pixelIdx := 1; pixelIdx < len(record) && pixelIdx <= height*width; pixelIdx++ {
			pixelValue, err := strconv.ParseFloat(record[pixelIdx], 32)
			if err != nil {
				return nil, fmt.Errorf("invalid pixel at row %d, col %d: %v", i, pixelIdx, err)
			}

			normalized := float32(pixelValue) / 255.0

			h := (pixelIdx - 1) / width
			w := (pixelIdx - 1) % width
			image.Data[0][h][w] = normalized
		}

		images[i-startIdx] = image
	}

	return &ImageDataset{
		Images:   images,
		Labels:   labels,
		Height:   height,
		Width:    width,
		Channels: channels,
	}, nil
}

func LoadBinaryImageFromCSV(filepath string, threshold float32) (*ImageDataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	hasHeader := false
	startIdx := 0
	if _, err := strconv.ParseFloat(records[0][0], 32); err != nil {
		hasHeader = true
		startIdx = 1
	}

	numSamples := len(records) - startIdx
	images := make([]*tensor.Tensor3D, numSamples)
	labels := make([][]float32, numSamples)

	height := 28
	width := 28
	channels := 1

	for i := startIdx; i < len(records); i++ {
		record := records[i]

		label, err := strconv.ParseFloat(record[0], 32)
		if err != nil {
			return nil, fmt.Errorf("invalid label at row %d: %v", i, err)
		}

		binaryLabel := float32(0)
		if label >= threshold {
			binaryLabel = 1
		}
		labels[i-startIdx] = []float32{binaryLabel}

		image := tensor.NewTensor3D(height, width, channels)

		for pixelIdx := 1; pixelIdx < len(record) && pixelIdx <= height*width; pixelIdx++ {
			pixelValue, err := strconv.ParseFloat(record[pixelIdx], 32)
			if err != nil {
				return nil, fmt.Errorf("invalid pixel at row %d, col %d: %v", i, pixelIdx, err)
			}

			normalized := float32(pixelValue) / 255.0

			h := (pixelIdx - 1) / width
			w := (pixelIdx - 1) % width
			image.Data[0][h][w] = normalized
		}

		images[i-startIdx] = image
	}

	return &ImageDataset{
		Images:   images,
		Labels:   labels,
		Height:   height,
		Width:    width,
		Channels: channels,
	}, nil
}

func SplitImageData(images []*tensor.Tensor3D, labels [][]float32, config SplitConfig) ImageSplit {
	numSamples := len(images)

	if config.Shuffle {
		images, labels = shuffleImageData(images, labels, config.Seed)
	}

	trainEnd := int(float64(numSamples) * config.TrainRatio)
	valEnd := trainEnd + int(float64(numSamples)*config.ValRatio)

	return ImageSplit{
		TrainX: images[:trainEnd],
		TrainY: labels[:trainEnd],
		ValX:   images[trainEnd:valEnd],
		ValY:   labels[trainEnd:valEnd],
		TestX:  images[valEnd:],
		TestY:  labels[valEnd:],
	}
}

type ImageSplit struct {
	TrainX []*tensor.Tensor3D
	TrainY [][]float32
	ValX   []*tensor.Tensor3D
	ValY   [][]float32
	TestX  []*tensor.Tensor3D
	TestY  [][]float32
}

func shuffleImageData(images []*tensor.Tensor3D, labels [][]float32, seed int64) ([]*tensor.Tensor3D, [][]float32) {
	numSamples := len(images)
	shuffledImages := make([]*tensor.Tensor3D, numSamples)
	shuffledLabels := make([][]float32, numSamples)

	indices := make([]int, numSamples)
	for i := range indices {
		indices[i] = i
	}

	rng := NewRNG(seed)
	for i := numSamples - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}

	for i, idx := range indices {
		shuffledImages[i] = images[idx]
		shuffledLabels[i] = labels[idx]
	}

	return shuffledImages, shuffledLabels
}
