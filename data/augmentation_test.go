package data

import "testing"

func TestFlipHorizontalMirrorsWidthAndKeepsOriginal(t *testing.T) {
	img := NewImage(2, 3, 1)
	img.Data[0][0][0], img.Data[0][1][0], img.Data[0][2][0] = 1, 2, 3
	img.Data[1][0][0], img.Data[1][1][0], img.Data[1][2][0] = 4, 5, 6

	flipped := FlipHorizontal(img)

	want := [][]float32{{3, 2, 1}, {6, 5, 4}}
	for h := 0; h < 2; h++ {
		for w := 0; w < 3; w++ {
			if got := flipped.Data[h][w][0]; got != want[h][w] {
				t.Errorf("flipped[%d][%d] = %v, want %v", h, w, got, want[h][w])
			}
		}
	}
	if img.Data[0][0][0] != 1 || img.Data[0][2][0] != 3 {
		t.Error("FlipHorizontal mutated its input image")
	}
}

func TestAugmentWithFlipsDoublesAndPairsLabels(t *testing.T) {
	a := NewImage(1, 2, 1)
	a.Data[0][0][0], a.Data[0][1][0] = 1, 2
	b := NewImage(1, 2, 1)
	b.Data[0][0][0], b.Data[0][1][0] = 3, 4
	labels := [][]float32{{1, 0}, {0, 1}}

	outImgs, outLabels := AugmentWithFlips([]*Image{a, b}, labels)

	if len(outImgs) != 4 || len(outLabels) != 4 {
		t.Fatalf("got %d images, %d labels, want 4 and 4", len(outImgs), len(outLabels))
	}
	// order: original, its flip, next original, its flip
	if outImgs[0] != a || outImgs[2] != b {
		t.Error("originals not at even indices")
	}
	if outImgs[1].Data[0][0][0] != 2 || outImgs[3].Data[0][0][0] != 4 {
		t.Error("flips not at odd indices or not mirrored")
	}
	for i, wantLabel := range [][]float32{{1, 0}, {1, 0}, {0, 1}, {0, 1}} {
		for j := range wantLabel {
			if outLabels[i][j] != wantLabel[j] {
				t.Errorf("label[%d] = %v, want %v", i, outLabels[i], wantLabel)
			}
		}
	}
}
