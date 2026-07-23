// data/imagefolder.go
package data

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
)

// LoadImageFolder walks root, treating each immediate subdirectory as one
// class (the standard ImageNet-style layout: root/classA/*.jpg,
// root/classB/*.png, ...), decoding every .jpg/.jpeg/.png file inside via
// the standard library's image/jpeg and image/png decoders (still
// dependency-free — both live in Go's standard library, unlike
// data/cifar10.go's raw binary format, this is the first loader in the
// package that needs a real image codec). Class order — and each class's
// one-hot index in the returned labels — is the subdirectories' sorted
// name order; that order is returned alongside the dataset so callers can
// map an index back to its class name. Every image must decode to the
// same height/width as the first one loaded.
//
// The directory walk is sequential (cheap: just stat calls), but the
// actual image.Decode calls — the expensive part for any real dataset —
// run across a worker pool bounded by GOMAXPROCS. Results are collected
// into an order-indexed slice first and then validated/assembled in a
// second, strictly sequential pass, so the returned dataset's order and
// every error message (including which path is reported as mismatched,
// always the first one in class/directory order to fail) are identical
// to a fully sequential load.
func LoadImageFolder(root string) (*ImageDataset, []string, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, nil, fmt.Errorf("data: LoadImageFolder: %w", err)
	}

	var classNames []string
	for _, e := range entries {
		if e.IsDir() {
			classNames = append(classNames, e.Name())
		}
	}
	sort.Strings(classNames)
	if len(classNames) == 0 {
		return nil, nil, fmt.Errorf("data: LoadImageFolder: %s has no subdirectories (expected one per class)", root)
	}

	type imgTask struct {
		classIdx int
		path     string
	}
	var tasks []imgTask
	for classIdx, name := range classNames {
		dir := filepath.Join(root, name)
		files, err := os.ReadDir(dir)
		if err != nil {
			return nil, nil, fmt.Errorf("data: LoadImageFolder: %w", err)
		}
		for _, f := range files {
			if f.IsDir() {
				continue
			}
			ext := strings.ToLower(filepath.Ext(f.Name()))
			if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
				continue
			}
			tasks = append(tasks, imgTask{classIdx, filepath.Join(dir, f.Name())})
		}
	}
	if len(tasks) == 0 {
		return nil, nil, fmt.Errorf("data: LoadImageFolder: %s contains no .jpg/.jpeg/.png files under its class subdirectories", root)
	}

	decoded := make([]*Image, len(tasks))
	decodeErrs := make([]error, len(tasks))

	workers := runtime.GOMAXPROCS(0)
	if workers > len(tasks) {
		workers = len(tasks)
	}
	if workers < 1 {
		workers = 1
	}

	taskChan := make(chan int, len(tasks))
	for i := range tasks {
		taskChan <- i
	}
	close(taskChan)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range taskChan {
				path := tasks[i].path
				file, err := os.Open(path)
				if err != nil {
					decodeErrs[i] = fmt.Errorf("data: LoadImageFolder: %w", err)
					continue
				}
				src, _, err := image.Decode(file)
				file.Close()
				if err != nil {
					decodeErrs[i] = fmt.Errorf("data: LoadImageFolder: decoding %s: %w", path, err)
					continue
				}
				decoded[i] = imageFromStdlib(src)
			}
		}()
	}
	wg.Wait()

	for _, err := range decodeErrs {
		if err != nil {
			return nil, nil, err
		}
	}

	var images []*Image
	var labels [][]float32
	height, width, channels := 0, 0, 0
	for i, t := range tasks {
		converted := decoded[i]
		if height == 0 {
			height, width, channels = converted.Height, converted.Width, converted.Channels
		} else if converted.Height != height || converted.Width != width {
			return nil, nil, fmt.Errorf("data: LoadImageFolder: %s has size %dx%d, want %dx%d (every image must match)", t.path, converted.Height, converted.Width, height, width)
		}

		images = append(images, converted)
		label := make([]float32, len(classNames))
		label[t.classIdx] = 1
		labels = append(labels, label)
	}

	return &ImageDataset{Images: images, Labels: labels, Height: height, Width: width, Channels: channels}, classNames, nil
}

// imageFromStdlib converts a decoded stdlib image.Image into this
// package's channel-last *Image, always as 3-channel RGB normalized to
// [0,1] — At(...).RGBA() returns equal r/g/b for a grayscale source, so
// this uniformly handles both without a separate grayscale path, and
// keeps every loaded image the same channel count regardless of its
// original file's color mode.
func imageFromStdlib(src image.Image) *Image {
	bounds := src.Bounds()
	h, w := bounds.Dy(), bounds.Dx()
	out := NewImage(h, w, 3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			out.Data[y][x][0] = float32(r) / 65535
			out.Data[y][x][1] = float32(g) / 65535
			out.Data[y][x][2] = float32(b) / 65535
		}
	}
	return out
}
