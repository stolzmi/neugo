package data

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

// Image is a package-local, channel-last (matching nn.Tensor's [batch, h,
// w, c] convention) image representation. data cannot import nn (design
// doc §4.1), so this exists instead of the deleted tensor.Tensor3D —
// callers that also import nn (i.e. examples/, never data itself) stack a
// []*Image into a batched *nn.Tensor.
type Image struct {
	Data                    [][][]float32 // [height][width][channel]
	Height, Width, Channels int
}

func NewImage(height, width, channels int) *Image {
	data := make([][][]float32, height)
	for h := range data {
		data[h] = make([][]float32, width)
		for w := range data[h] {
			data[h][w] = make([]float32, channels)
		}
	}
	return &Image{Data: data, Height: height, Width: width, Channels: channels}
}

type ImageDataset struct {
	Images   []*Image
	Labels   [][]float32
	Height   int
	Width    int
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

	startIdx := 0
	if _, err := strconv.ParseFloat(records[0][0], 32); err != nil {
		startIdx = 1
	}

	numSamples := len(records) - startIdx
	images := make([]*Image, numSamples)
	labels := make([][]float32, numSamples)
	height, width, channels := 28, 28, 1

	for i := startIdx; i < len(records); i++ {
		record := records[i]
		label, err := strconv.ParseFloat(record[0], 32)
		if err != nil {
			return nil, fmt.Errorf("invalid label at row %d: %v", i, err)
		}
		labels[i-startIdx] = make([]float32, 10)
		labels[i-startIdx][int(label)] = 1.0

		img := NewImage(height, width, channels)
		for pixelIdx := 1; pixelIdx < len(record) && pixelIdx <= height*width; pixelIdx++ {
			pixelValue, err := strconv.ParseFloat(record[pixelIdx], 32)
			if err != nil {
				return nil, fmt.Errorf("invalid pixel at row %d, col %d: %v", i, pixelIdx, err)
			}
			h := (pixelIdx - 1) / width
			w := (pixelIdx - 1) % width
			img.Data[h][w][0] = float32(pixelValue) / 255.0
		}
		images[i-startIdx] = img
	}

	return &ImageDataset{Images: images, Labels: labels, Height: height, Width: width, Channels: channels}, nil
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

	startIdx := 0
	if _, err := strconv.ParseFloat(records[0][0], 32); err != nil {
		startIdx = 1
	}

	numSamples := len(records) - startIdx
	images := make([]*Image, numSamples)
	labels := make([][]float32, numSamples)
	height, width, channels := 28, 28, 1

	for i := startIdx; i < len(records); i++ {
		record := records[i]
		label, err := strconv.ParseFloat(record[0], 32)
		if err != nil {
			return nil, fmt.Errorf("invalid label at row %d: %v", i, err)
		}
		binaryLabel := float32(0)
		if float32(label) >= threshold { // was `label >= threshold`: float64 vs float32, would not compile
			binaryLabel = 1
		}
		labels[i-startIdx] = []float32{binaryLabel}

		img := NewImage(height, width, channels)
		for pixelIdx := 1; pixelIdx < len(record) && pixelIdx <= height*width; pixelIdx++ {
			pixelValue, err := strconv.ParseFloat(record[pixelIdx], 32)
			if err != nil {
				return nil, fmt.Errorf("invalid pixel at row %d, col %d: %v", i, pixelIdx, err)
			}
			h := (pixelIdx - 1) / width
			w := (pixelIdx - 1) % width
			img.Data[h][w][0] = float32(pixelValue) / 255.0
		}
		images[i-startIdx] = img
	}

	return &ImageDataset{Images: images, Labels: labels, Height: height, Width: width, Channels: channels}, nil
}

type ImageSplit struct {
	TrainX []*Image
	TrainY [][]float32
	ValX   []*Image
	ValY   [][]float32
	TestX  []*Image
	TestY  [][]float32
}

func SplitImageData(rng *rand.Rand, images []*Image, labels [][]float32, config SplitConfig) ImageSplit {
	numSamples := len(images)
	if config.Shuffle {
		images, labels = shuffleImageData(rng, images, labels)
	}
	trainEnd := int(float64(numSamples) * config.TrainRatio)
	valEnd := trainEnd + int(float64(numSamples)*config.ValRatio)
	return ImageSplit{
		TrainX: images[:trainEnd], TrainY: labels[:trainEnd],
		ValX: images[trainEnd:valEnd], ValY: labels[trainEnd:valEnd],
		TestX: images[valEnd:], TestY: labels[valEnd:],
	}
}

func shuffleImageData(rng *rand.Rand, images []*Image, labels [][]float32) ([]*Image, [][]float32) {
	n := len(images)
	shuffledImages := make([]*Image, n)
	shuffledLabels := make([][]float32, n)
	for i, idx := range rng.Perm(n) {
		shuffledImages[i] = images[idx]
		shuffledLabels[i] = labels[idx]
	}
	return shuffledImages, shuffledLabels
}
