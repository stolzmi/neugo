package data

import "testing"

func TestNewImageShape(t *testing.T) {
	img := NewImage(4, 5, 3)
	if img.Height != 4 || img.Width != 5 || img.Channels != 3 {
		t.Fatalf("NewImage shape = (%d,%d,%d), want (4,5,3)", img.Height, img.Width, img.Channels)
	}
	if len(img.Data) != 4 || len(img.Data[0]) != 5 || len(img.Data[0][0]) != 3 {
		t.Fatalf("Data dims = (%d,%d,%d), want (4,5,3)", len(img.Data), len(img.Data[0]), len(img.Data[0][0]))
	}
}

func TestSplitImageDataRatios(t *testing.T) {
	images := make([]*Image, 10)
	labels := make([][]float32, 10)
	for i := range images {
		images[i] = NewImage(2, 2, 1)
		labels[i] = []float32{float32(i)}
	}
	rng := newTestRNG(1)
	split := SplitImageData(rng, images, labels, SplitConfig{TrainRatio: 0.6, ValRatio: 0.2, TestRatio: 0.2, Shuffle: true})
	if len(split.TrainX) != 6 || len(split.ValX) != 2 || len(split.TestX) != 2 {
		t.Fatalf("split sizes = (%d,%d,%d), want (6,2,2)", len(split.TrainX), len(split.ValX), len(split.TestX))
	}
}
