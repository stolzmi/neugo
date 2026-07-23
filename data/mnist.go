package data

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// MNISTDataset holds MNIST image and label data.
type MNISTDataset struct {
	Images     []*Image
	Labels     [][]float32
	ClassNames []string
}

// LoadMNISTImages loads MNIST image data from IDX3-UBYTE format.
func LoadMNISTImages(filepath string) ([]*Image, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	var magic, numImages, rows, cols int32
	if err := binary.Read(file, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != 2051 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}

	if err := binary.Read(file, binary.BigEndian, &numImages); err != nil {
		return nil, fmt.Errorf("read num images: %w", err)
	}
	if err := binary.Read(file, binary.BigEndian, &rows); err != nil {
		return nil, fmt.Errorf("read rows: %w", err)
	}
	if err := binary.Read(file, binary.BigEndian, &cols); err != nil {
		return nil, fmt.Errorf("read cols: %w", err)
	}

	images := make([]*Image, numImages)
	pixelData := make([]byte, rows*cols)

	for i := 0; i < int(numImages); i++ {
		_, err := io.ReadFull(file, pixelData)
		if err != nil {
			return nil, fmt.Errorf("read image %d: %w", i, err)
		}

		img := NewImage(int(rows), int(cols), 1)
		for h := 0; h < int(rows); h++ {
			for w := 0; w < int(cols); w++ {
				val := float32(pixelData[h*int(cols)+w]) / 255.0
				img.Data[h][w][0] = val
			}
		}
		images[i] = img
	}

	return images, nil
}

// LoadMNISTLabels loads MNIST label data from IDX1-UBYTE format.
func LoadMNISTLabels(filepath string) ([][]float32, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	var magic, numLabels int32
	if err := binary.Read(file, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}

	if err := binary.Read(file, binary.BigEndian, &numLabels); err != nil {
		return nil, fmt.Errorf("read num labels: %w", err)
	}

	labels := make([][]float32, numLabels)
	labelData := make([]byte, numLabels)

	_, err = io.ReadFull(file, labelData)
	if err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}

	for i := 0; i < int(numLabels); i++ {
		oneHot := make([]float32, 10)
		oneHot[labelData[i]] = 1.0
		labels[i] = oneHot
	}

	return labels, nil
}

// LoadMNIST loads both images and labels from MNIST binary format.
func LoadMNIST(imagesPath, labelsPath string) (*MNISTDataset, error) {
	images, err := LoadMNISTImages(imagesPath)
	if err != nil {
		return nil, fmt.Errorf("load images: %w", err)
	}

	labels, err := LoadMNISTLabels(labelsPath)
	if err != nil {
		return nil, fmt.Errorf("load labels: %w", err)
	}

	if len(images) != len(labels) {
		return nil, fmt.Errorf("image/label count mismatch: %d vs %d", len(images), len(labels))
	}

	classNames := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
	return &MNISTDataset{
		Images:     images,
		Labels:     labels,
		ClassNames: classNames,
	}, nil
}
