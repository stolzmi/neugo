package data

import (
	"math"
	"testing"
)

func TestNormalizeImagesWithStatsReturnsZeroMeanUnitStd(t *testing.T) {
	images := make([]*Image, 4)
	for i := range images {
		img := NewImage(2, 2, 1)
		v := float32(i) * 2.0
		for h := 0; h < 2; h++ {
			for w := 0; w < 2; w++ {
				img.Data[h][w][0] = v
			}
		}
		images[i] = img
	}
	normalized, stats := NormalizeImagesWithStats(images)

	if len(stats.Mean) != 1 || len(stats.Std) != 1 {
		t.Fatalf("stats has %d means / %d stds, want 1 and 1", len(stats.Mean), len(stats.Std))
	}
	wantMean := float32(3.0) // mean of 0,2,4,6
	if diff := math.Abs(float64(stats.Mean[0] - wantMean)); diff > 1e-4 {
		t.Errorf("Mean[0] = %v, want %v", stats.Mean[0], wantMean)
	}

	var sum, sumSq float32
	for _, img := range normalized {
		v := img.Data[0][0][0]
		sum += v
		sumSq += v * v
	}
	n := float32(len(normalized))
	mean, variance := sum/n, sumSq/n-((sum/n)*(sum/n))
	if math.Abs(float64(mean)) > 1e-4 {
		t.Errorf("normalized mean = %v, want ~0", mean)
	}
	if diff := math.Abs(float64(variance - 1)); diff > 1e-3 {
		t.Errorf("normalized variance = %v, want ~1", variance)
	}
}

func TestNormalizeImagesMatchesWithStatsOutput(t *testing.T) {
	images := []*Image{NewImage(1, 1, 1), NewImage(1, 1, 1)}
	images[0].Data[0][0][0] = 1
	images[1].Data[0][0][0] = 5

	viaWrapper := NormalizeImages(images)
	viaWithStats, _ := NormalizeImagesWithStats(images)

	for i := range viaWrapper {
		if viaWrapper[i].Data[0][0][0] != viaWithStats[i].Data[0][0][0] {
			t.Errorf("image %d: NormalizeImages = %v, NormalizeImagesWithStats = %v", i, viaWrapper[i].Data[0][0][0], viaWithStats[i].Data[0][0][0])
		}
	}
}
