package data

import (
	"math/rand"
	"testing"
)

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

func TestBilinearSampleExactPixelsAndMidpoint(t *testing.T) {
	img := NewImage(2, 2, 1)
	img.Data[0][0][0], img.Data[0][1][0] = 1, 2
	img.Data[1][0][0], img.Data[1][1][0] = 3, 4

	if got := bilinearSample(img, 0, 0); got[0] != 1 {
		t.Errorf("bilinearSample(0,0) = %v, want 1", got[0])
	}
	got := bilinearSample(img, 0.5, 0.5)
	want := float32(1+2+3+4) / 4
	if diff := got[0] - want; diff > 1e-4 || diff < -1e-4 {
		t.Errorf("bilinearSample(0.5,0.5) = %v, want %v (average of all 4)", got[0], want)
	}
}

func TestBilinearSampleFarOutOfBoundsReturnsZero(t *testing.T) {
	img := NewImage(2, 2, 1)
	img.Data[0][0][0] = 5
	got := bilinearSample(img, -5, -5)
	if got[0] != 0 {
		t.Errorf("bilinearSample far out of bounds = %v, want 0", got[0])
	}
}

func TestRandomRotateZeroDegreesIsIdentity(t *testing.T) {
	img := NewImage(3, 3, 1)
	val := float32(1)
	for h := range img.Data {
		for w := range img.Data[h] {
			img.Data[h][w][0] = val
			val++
		}
	}
	rng := rand.New(rand.NewSource(1))
	out := RandomRotate(rng, img, 0)
	for h := range img.Data {
		for w := range img.Data[h] {
			if diff := out.Data[h][w][0] - img.Data[h][w][0]; diff > 1e-4 || diff < -1e-4 {
				t.Errorf("out[%d][%d] = %v, want %v", h, w, out.Data[h][w][0], img.Data[h][w][0])
			}
		}
	}
}

func TestRandomRotatePreservesShape(t *testing.T) {
	img := NewImage(5, 7, 3)
	rng := rand.New(rand.NewSource(1))
	out := RandomRotate(rng, img, 15)
	if out.Height != 5 || out.Width != 7 || out.Channels != 3 {
		t.Fatalf("shape = (%d,%d,%d), want (5,7,3)", out.Height, out.Width, out.Channels)
	}
}

func TestRandomCropReturnsContiguousWindowOfOriginal(t *testing.T) {
	img := NewImage(4, 4, 1)
	val := float32(0)
	for h := 0; h < 4; h++ {
		for w := 0; w < 4; w++ {
			img.Data[h][w][0] = val
			val++
		}
	}
	rng := rand.New(rand.NewSource(1))
	cropped := RandomCrop(rng, img, 2, 2, 0)

	found := false
	for top := 0; top <= 2 && !found; top++ {
		for left := 0; left <= 2 && !found; left++ {
			match := true
			for h := 0; h < 2 && match; h++ {
				for w := 0; w < 2 && match; w++ {
					if cropped.Data[h][w][0] != img.Data[top+h][left+w][0] {
						match = false
					}
				}
			}
			if match {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("RandomCrop output is not a contiguous window of the original image")
	}
}

func TestRandomCropWithPaddingZeroPadsBorder(t *testing.T) {
	img := NewImage(2, 2, 1)
	img.Data[0][0][0], img.Data[0][1][0] = 1, 2
	img.Data[1][0][0], img.Data[1][1][0] = 3, 4
	rng := rand.New(rand.NewSource(1))
	// padding=1, cropH=cropW=4 (the padded size exactly) -> top=left=0
	// deterministically, since maxTop=maxLeft=0 skips the random draw.
	out := RandomCrop(rng, img, 4, 4, 1)
	if out.Data[0][0][0] != 0 || out.Data[0][3][0] != 0 || out.Data[3][0][0] != 0 || out.Data[3][3][0] != 0 {
		t.Error("expected zero-padded border")
	}
	if out.Data[1][1][0] != 1 || out.Data[1][2][0] != 2 || out.Data[2][1][0] != 3 || out.Data[2][2][0] != 4 {
		t.Error("expected original image values at the center")
	}
}

func TestColorJitterZeroStrengthIsIdentity(t *testing.T) {
	img := NewImage(2, 2, 3)
	val := float32(0.1)
	for h := range img.Data {
		for w := range img.Data[h] {
			for c := range img.Data[h][w] {
				img.Data[h][w][c] = val
				val += 0.05
			}
		}
	}
	rng := rand.New(rand.NewSource(1))
	out := ColorJitter(rng, img, 0, 0, 0)
	for h := range img.Data {
		for w := range img.Data[h] {
			for c := range img.Data[h][w] {
				if out.Data[h][w][c] != img.Data[h][w][c] {
					t.Errorf("out[%d][%d][%d] = %v, want %v (identity)", h, w, c, out.Data[h][w][c], img.Data[h][w][c])
				}
			}
		}
	}
}

func TestColorJitterBrightnessOnlyScalesUniformly(t *testing.T) {
	img := NewImage(2, 2, 1)
	val := float32(1)
	for h := range img.Data {
		for w := range img.Data[h] {
			img.Data[h][w][0] = val
			val++
		}
	}
	rng := rand.New(rand.NewSource(1))
	out := ColorJitter(rng, img, 0.5, 0, 0)
	ratio := out.Data[0][0][0] / img.Data[0][0][0]
	for h := range img.Data {
		for w := range img.Data[h] {
			got := out.Data[h][w][0] / img.Data[h][w][0]
			if diff := got - ratio; diff > 1e-4 || diff < -1e-4 {
				t.Errorf("brightness-only ratio at [%d][%d] = %v, want uniform %v", h, w, got, ratio)
			}
		}
	}
}

func TestColorJitterSkipsSaturationForNonRGB(t *testing.T) {
	// Channels=1: saturation must be a no-op regardless of its strength,
	// since there's no color to desaturate.
	img := NewImage(2, 2, 1)
	img.Data[0][0][0] = 0.5
	rng := rand.New(rand.NewSource(1))
	out := ColorJitter(rng, img, 0, 0, 1.0)
	if out.Data[0][0][0] != 0.5 {
		t.Errorf("saturation modified a single-channel image: got %v, want unchanged 0.5", out.Data[0][0][0])
	}
}

func TestCutoutZeroSizeIsIdentity(t *testing.T) {
	img := NewImage(2, 2, 1)
	img.Data[0][0][0] = 5
	rng := rand.New(rand.NewSource(1))
	out := Cutout(rng, img, 0)
	if out.Data[0][0][0] != 5 {
		t.Error("Cutout with size=0 should not modify the image")
	}
}

func TestCutoutLargeSizeZeroesEntireImage(t *testing.T) {
	img := NewImage(4, 4, 1)
	for h := range img.Data {
		for w := range img.Data[h] {
			img.Data[h][w][0] = 7
		}
	}
	rng := rand.New(rand.NewSource(2))
	// size far larger than the image guarantees full coverage regardless
	// of which pixel the random center lands on.
	out := Cutout(rng, img, 100)
	for h := range out.Data {
		for w := range out.Data[h] {
			if out.Data[h][w][0] != 0 {
				t.Errorf("out.Data[%d][%d][0] = %v, want 0", h, w, out.Data[h][w][0])
			}
		}
	}
}
