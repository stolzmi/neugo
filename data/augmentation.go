// data/augmentation.go
package data

import (
	"math"
	"math/rand"
)

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

// bilinearSample reads img at fractional coordinate (y, x), bilinearly
// interpolating between the 4 nearest pixels; any of the 4 that falls
// outside the image contributes 0 (zero padding), and a query more than a
// full pixel outside the image on every axis returns all zeros directly —
// used by RandomRotate to sample the source image at a rotated position
// that may land outside its bounds.
func bilinearSample(img *Image, y, x float64) []float32 {
	out := make([]float32, img.Channels)
	if x < -1 || x > float64(img.Width) || y < -1 || y > float64(img.Height) {
		return out
	}
	x0, y0 := int(math.Floor(x)), int(math.Floor(y))
	x1, y1 := x0+1, y0+1
	fx, fy := float32(x-float64(x0)), float32(y-float64(y0))

	get := func(yy, xx int) []float32 {
		if yy < 0 || yy >= img.Height || xx < 0 || xx >= img.Width {
			return nil
		}
		return img.Data[yy][xx]
	}
	p00, p01, p10, p11 := get(y0, x0), get(y0, x1), get(y1, x0), get(y1, x1)
	for c := 0; c < img.Channels; c++ {
		var v00, v01, v10, v11 float32
		if p00 != nil {
			v00 = p00[c]
		}
		if p01 != nil {
			v01 = p01[c]
		}
		if p10 != nil {
			v10 = p10[c]
		}
		if p11 != nil {
			v11 = p11[c]
		}
		top := v00*(1-fx) + v01*fx
		bottom := v10*(1-fx) + v11*fx
		out[c] = top*(1-fy) + bottom*fy
	}
	return out
}

// RandomRotate rotates img by a uniformly random angle in
// [-maxDegrees, maxDegrees] around its center, bilinearly resampling so
// the output stays the same size as the input; pixels rotated in from
// outside the original bounds are filled with 0.
func RandomRotate(rng *rand.Rand, img *Image, maxDegrees float64) *Image {
	angleDeg := (rng.Float64()*2 - 1) * maxDegrees
	rad := angleDeg * math.Pi / 180
	cosA, sinA := math.Cos(rad), math.Sin(rad)
	cx := float64(img.Width-1) / 2
	cy := float64(img.Height-1) / 2

	out := NewImage(img.Height, img.Width, img.Channels)
	for oh := 0; oh < img.Height; oh++ {
		for ow := 0; ow < img.Width; ow++ {
			dx := float64(ow) - cx
			dy := float64(oh) - cy
			// Sampling the source at the *inverse* rotation of the output
			// coordinate is the standard backward-warp approach — it
			// guarantees every output pixel gets a value (possibly 0),
			// unlike forward-warping each input pixel into the output.
			srcX := cx + dx*cosA + dy*sinA
			srcY := cy - dx*sinA + dy*cosA
			copy(out.Data[oh][ow], bilinearSample(img, srcY, srcX))
		}
	}
	return out
}

// RandomCrop zero-pads img by padding pixels on every side, then takes a
// uniformly random cropH x cropW window from the padded image — the
// standard CIFAR-style "pad and crop" augmentation. cropH/cropW must not
// exceed the padded image's height/width.
func RandomCrop(rng *rand.Rand, img *Image, cropH, cropW, padding int) *Image {
	paddedH, paddedW := img.Height+2*padding, img.Width+2*padding
	padded := NewImage(paddedH, paddedW, img.Channels)
	for h := 0; h < img.Height; h++ {
		for w := 0; w < img.Width; w++ {
			copy(padded.Data[h+padding][w+padding], img.Data[h][w])
		}
	}

	top, left := 0, 0
	if maxTop := paddedH - cropH; maxTop > 0 {
		top = rng.Intn(maxTop + 1)
	}
	if maxLeft := paddedW - cropW; maxLeft > 0 {
		left = rng.Intn(maxLeft + 1)
	}

	out := NewImage(cropH, cropW, img.Channels)
	for h := 0; h < cropH; h++ {
		for w := 0; w < cropW; w++ {
			copy(out.Data[h][w], padded.Data[top+h][left+w])
		}
	}
	return out
}

// ColorJitter randomly perturbs brightness, contrast, and (for 3-channel/
// RGB images only) saturation, each by an independent factor drawn
// uniformly from [1-strength, 1+strength] for that adjustment's strength
// argument (0 disables that adjustment entirely). Contrast is applied
// around the image's own mean pixel value; saturation is applied around
// each pixel's luminance (standard RGB -> grayscale weights).
func ColorJitter(rng *rand.Rand, img *Image, brightness, contrast, saturation float32) *Image {
	out := NewImage(img.Height, img.Width, img.Channels)
	for h := range img.Data {
		for w := range img.Data[h] {
			copy(out.Data[h][w], img.Data[h][w])
		}
	}

	if brightness > 0 {
		delta := 1 + (rng.Float32()*2-1)*brightness
		for h := range out.Data {
			for w := range out.Data[h] {
				for c := range out.Data[h][w] {
					out.Data[h][w][c] *= delta
				}
			}
		}
	}

	if contrast > 0 {
		var sum float64
		count := 0
		for h := range out.Data {
			for w := range out.Data[h] {
				for _, v := range out.Data[h][w] {
					sum += float64(v)
					count++
				}
			}
		}
		mean := float32(sum / float64(count))
		delta := 1 + (rng.Float32()*2-1)*contrast
		for h := range out.Data {
			for w := range out.Data[h] {
				for c := range out.Data[h][w] {
					out.Data[h][w][c] = mean + (out.Data[h][w][c]-mean)*delta
				}
			}
		}
	}

	if saturation > 0 && img.Channels == 3 {
		delta := 1 + (rng.Float32()*2-1)*saturation
		for h := range out.Data {
			for w := range out.Data[h] {
				px := out.Data[h][w]
				gray := 0.299*px[0] + 0.587*px[1] + 0.114*px[2]
				for c := 0; c < 3; c++ {
					px[c] = gray + (px[c]-gray)*delta
				}
			}
		}
	}

	return out
}

// Cutout zeroes out a random size x size square patch of img (Devries &
// Taylor, 2017) — the patch is centered at a uniformly random pixel and
// clipped at the image boundary, so it can partially fall outside.
func Cutout(rng *rand.Rand, img *Image, size int) *Image {
	out := NewImage(img.Height, img.Width, img.Channels)
	for h := range img.Data {
		for w := range img.Data[h] {
			copy(out.Data[h][w], img.Data[h][w])
		}
	}
	if size <= 0 {
		return out
	}
	cy, cx := rng.Intn(img.Height), rng.Intn(img.Width)
	half := size / 2
	top, left := cy-half, cx-half
	for h := top; h < top+size; h++ {
		if h < 0 || h >= img.Height {
			continue
		}
		for w := left; w < left+size; w++ {
			if w < 0 || w >= img.Width {
				continue
			}
			for c := range out.Data[h][w] {
				out.Data[h][w][c] = 0
			}
		}
	}
	return out
}
