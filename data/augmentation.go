// data/augmentation.go
package data

// FlipHorizontal returns a new Image mirrored left-to-right; img is left
// unmodified.
func FlipHorizontal(img *Image) *Image {
	out := NewImage(img.Height, img.Width, img.Channels)
	for h := 0; h < img.Height; h++ {
		for w := 0; w < img.Width; w++ {
			copy(out.Data[h][w], img.Data[h][img.Width-1-w])
		}
	}
	return out
}

// AugmentWithFlips doubles a dataset by interleaving each image with its
// horizontal mirror (img0, flip0, img1, flip1, ...), duplicating each
// label for its flipped pair. A standard, cheap augmentation for datasets
// without meaningful left-right asymmetry (e.g. CIFAR-10/100).
func AugmentWithFlips(images []*Image, labels [][]float32) ([]*Image, [][]float32) {
	outImages := make([]*Image, 0, len(images)*2)
	outLabels := make([][]float32, 0, len(labels)*2)
	for i, img := range images {
		outImages = append(outImages, img, FlipHorizontal(img))
		outLabels = append(outLabels, labels[i], labels[i])
	}
	return outImages, outLabels
}
