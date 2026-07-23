package data

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func writeBenchPNG(b *testing.B, path string, w, h int, c color.RGBA) {
	b.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		b.Fatalf("create %s: %v", path, err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		b.Fatalf("encode %s: %v", path, err)
	}
}

// BenchmarkLoadImageFolder decodes a synthetic 200-image, 4-class dataset
// (64x64 PNGs) to measure the worker-pool decode's throughput. Unlike the
// optimizer benchmarks in train/bench_test.go, there's no in-process
// sequential/parallel toggle here (LoadImageFolder doesn't depend on
// nn.SetDeterministic) — compare wall time across GOMAXPROCS settings
// directly to see the parallel decode's effect:
//
//	go test ./data/ -bench BenchmarkLoadImageFolder -benchtime 5x -run ^$
//	GOMAXPROCS=1 go test ./data/ -bench BenchmarkLoadImageFolder -benchtime 5x -run ^$
func BenchmarkLoadImageFolder(b *testing.B) {
	root := b.TempDir()
	classNames := []string{"a", "b", "c", "d"}
	for _, name := range classNames {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0755); err != nil {
			b.Fatal(err)
		}
		for i := 0; i < 50; i++ {
			path := filepath.Join(dir, fmt.Sprintf("%03d.png", i))
			shade := uint8(i * 5)
			writeBenchPNG(b, path, 64, 64, color.RGBA{shade, shade, shade, 255})
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := LoadImageFolder(root); err != nil {
			b.Fatal(err)
		}
	}
}
