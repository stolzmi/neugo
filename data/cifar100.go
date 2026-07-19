package data

import (
	"bufio"
	"io"
	"os"
	"strings"
)

// CIFAR100Dataset holds CIFAR-100 images with both label granularities:
// FineLabels is 100-way one-hot (the usual classification target),
// CoarseLabels is 20-way one-hot (the superclass each fine label belongs
// to). ClassNames, when populated via LoadCIFAR100ClassNames, holds the
// fine label names in label-index order; CoarseClassNames holds the
// coarse (superclass) names.
type CIFAR100Dataset struct {
	Images           []*Image
	FineLabels       [][]float32
	CoarseLabels     [][]float32
	ClassNames       []string
	CoarseClassNames []string
}

// LoadCIFAR100Binary reads one CIFAR-100 binary file (train.bin or
// test.bin from the official cifar-100-binary distribution). Each record
// is 1 coarse-label byte + 1 fine-label byte + 3072 pixel bytes (32x32
// RGB, red channel first) — 3074 bytes total, one byte longer per record
// than CIFAR-10's format because of the extra coarse-label byte.
func LoadCIFAR100Binary(filepath string) (*CIFAR100Dataset, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	images := make([]*Image, 0)
	fineLabels := make([][]float32, 0)
	coarseLabels := make([][]float32, 0)

	buffer := make([]byte, 3074)

	for {
		n, err := io.ReadFull(file, buffer)
		if err == io.EOF {
			break
		}
		if err != nil || n != 3074 {
			break
		}

		coarse := int(buffer[0])
		fine := int(buffer[1])

		coarseOneHot := make([]float32, 20)
		coarseOneHot[coarse] = 1.0
		coarseLabels = append(coarseLabels, coarseOneHot)

		fineOneHot := make([]float32, 100)
		fineOneHot[fine] = 1.0
		fineLabels = append(fineLabels, fineOneHot)

		img := NewImage(32, 32, 3)

		for c := 0; c < 3; c++ {
			for h := 0; h < 32; h++ {
				for w := 0; w < 32; w++ {
					idx := 2 + c*1024 + h*32 + w
					img.Data[h][w][c] = float32(buffer[idx]) / 255.0
				}
			}
		}

		images = append(images, img)
	}

	return &CIFAR100Dataset{
		Images:       images,
		FineLabels:   fineLabels,
		CoarseLabels: coarseLabels,
	}, nil
}

// LoadCIFAR100BinaryBatch concatenates multiple CIFAR-100 binary files
// (e.g. train.bin and test.bin) into one dataset.
func LoadCIFAR100BinaryBatch(filepaths []string) (*CIFAR100Dataset, error) {
	allImages := make([]*Image, 0)
	allFine := make([][]float32, 0)
	allCoarse := make([][]float32, 0)

	for _, filepath := range filepaths {
		dataset, err := LoadCIFAR100Binary(filepath)
		if err != nil {
			return nil, err
		}
		allImages = append(allImages, dataset.Images...)
		allFine = append(allFine, dataset.FineLabels...)
		allCoarse = append(allCoarse, dataset.CoarseLabels...)
	}

	return &CIFAR100Dataset{
		Images:       allImages,
		FineLabels:   allFine,
		CoarseLabels: allCoarse,
	}, nil
}

// LoadCIFAR100ClassNames reads the newline-separated label-name files that
// ship alongside the CIFAR-100 binary distribution (fine_label_names.txt,
// coarse_label_names.txt) — one name per line, in label-index order, so
// line N is the name for label N. Returns fine names (100) and coarse
// names (20).
func LoadCIFAR100ClassNames(fineNamesPath, coarseNamesPath string) (fine, coarse []string, err error) {
	fine, err = readLabelNames(fineNamesPath)
	if err != nil {
		return nil, nil, err
	}
	coarse, err = readLabelNames(coarseNamesPath)
	if err != nil {
		return nil, nil, err
	}
	return fine, coarse, nil
}

func readLabelNames(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var names []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			names = append(names, line)
		}
	}
	return names, scanner.Err()
}
