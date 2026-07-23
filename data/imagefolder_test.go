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

// writeTestPNG writes a solid-color w x h PNG to path, for building a
// synthetic ImageFolder-style dataset without depending on real image
// files being present in the repo.
func writeTestPNG(t *testing.T, path string, w, h int, c color.RGBA) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatalf("encode %s: %v", path, err)
	}
}

func TestLoadImageFolderTwoClasses(t *testing.T) {
	root := t.TempDir()
	catDir := filepath.Join(root, "cat")
	dogDir := filepath.Join(root, "dog")
	if err := os.MkdirAll(catDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dogDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestPNG(t, filepath.Join(catDir, "a.png"), 4, 4, color.RGBA{255, 0, 0, 255})
	writeTestPNG(t, filepath.Join(catDir, "b.png"), 4, 4, color.RGBA{200, 0, 0, 255})
	writeTestPNG(t, filepath.Join(dogDir, "c.png"), 4, 4, color.RGBA{0, 255, 0, 255})

	ds, classNames, err := LoadImageFolder(root)
	if err != nil {
		t.Fatalf("LoadImageFolder: %v", err)
	}

	if len(classNames) != 2 || classNames[0] != "cat" || classNames[1] != "dog" {
		t.Fatalf("classNames = %v, want [cat dog] (sorted)", classNames)
	}
	if len(ds.Images) != 3 || len(ds.Labels) != 3 {
		t.Fatalf("got %d images, %d labels, want 3 and 3", len(ds.Images), len(ds.Labels))
	}
	if ds.Height != 4 || ds.Width != 4 || ds.Channels != 3 {
		t.Fatalf("dataset shape = (%d,%d,%d), want (4,4,3)", ds.Height, ds.Width, ds.Channels)
	}

	// First two images are "cat" (class index 0), one-hot [1,0]; the
	// third is "dog" (class index 1), one-hot [0,1].
	for i := 0; i < 2; i++ {
		if ds.Labels[i][0] != 1 || ds.Labels[i][1] != 0 {
			t.Errorf("Labels[%d] = %v, want [1 0] (cat)", i, ds.Labels[i])
		}
	}
	if ds.Labels[2][0] != 0 || ds.Labels[2][1] != 1 {
		t.Errorf("Labels[2] = %v, want [0 1] (dog)", ds.Labels[2])
	}

	// Red channel should dominate for the red cat image.
	if r := ds.Images[0].Data[0][0][0]; r < 0.9 {
		t.Errorf("red pixel value = %v, want close to 1.0", r)
	}
}

func TestLoadImageFolderRejectsMismatchedSizes(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "onlyclass")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestPNG(t, filepath.Join(dir, "a.png"), 4, 4, color.RGBA{255, 255, 255, 255})
	writeTestPNG(t, filepath.Join(dir, "b.png"), 8, 8, color.RGBA{255, 255, 255, 255})

	if _, _, err := LoadImageFolder(root); err == nil {
		t.Fatal("expected error for mismatched image sizes, got nil")
	}
}

func TestLoadImageFolderRejectsNoSubdirectories(t *testing.T) {
	root := t.TempDir()
	if _, _, err := LoadImageFolder(root); err == nil {
		t.Fatal("expected error for a root with no class subdirectories, got nil")
	}
}

func TestLoadImageFolderManyFilesPreservesOrderUnderConcurrentDecode(t *testing.T) {
	// Decoding now runs across a worker pool; with enough files this
	// meaningfully exercises multiple goroutines racing to decode and
	// write into their order-indexed slot, so this checks that the final
	// dataset still comes out in the exact class-then-filename order a
	// sequential load would produce, with every label matching its class.
	root := t.TempDir()
	classNames := []string{"alpha", "beta", "gamma"}
	perClass := 20
	for _, name := range classNames {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < perClass; i++ {
			// Filenames sort numerically as strings only if
			// zero-padded; zero-pad so os.ReadDir's lexicographic
			// order matches the intended sequence index.
			path := filepath.Join(dir, filepathBase(i)+".png")
			shade := uint8(i * (255 / perClass))
			writeTestPNG(t, path, 2, 2, color.RGBA{shade, shade, shade, 255})
		}
	}

	ds, gotClassNames, err := LoadImageFolder(root)
	if err != nil {
		t.Fatalf("LoadImageFolder: %v", err)
	}
	for i, name := range classNames {
		if gotClassNames[i] != name {
			t.Fatalf("classNames = %v, want %v", gotClassNames, classNames)
		}
	}

	want := len(classNames) * perClass
	if len(ds.Images) != want || len(ds.Labels) != want {
		t.Fatalf("got %d images, %d labels, want %d and %d", len(ds.Images), len(ds.Labels), want, want)
	}

	for classIdx := range classNames {
		for i := 0; i < perClass; i++ {
			idx := classIdx*perClass + i
			label := ds.Labels[idx]
			for c := range label {
				want := float32(0)
				if c == classIdx {
					want = 1
				}
				if label[c] != want {
					t.Fatalf("Labels[%d] = %v, want one-hot at index %d", idx, label, classIdx)
				}
			}
			wantShade := float32(i*(255/perClass)) / 255
			if diff := ds.Images[idx].Data[0][0][0] - wantShade; diff > 0.02 || diff < -0.02 {
				t.Fatalf("Images[%d] shade = %v, want ~%v (order mismatch after concurrent decode?)", idx, ds.Images[idx].Data[0][0][0], wantShade)
			}
		}
	}
}

func filepathBase(i int) string {
	return fmt.Sprintf("%03d", i)
}

func TestLoadImageFolderIgnoresNonImageFiles(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "classA")
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestPNG(t, filepath.Join(dir, "img.png"), 2, 2, color.RGBA{1, 2, 3, 255})
	if err := os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("not an image"), 0644); err != nil {
		t.Fatal(err)
	}

	ds, _, err := LoadImageFolder(root)
	if err != nil {
		t.Fatalf("LoadImageFolder: %v", err)
	}
	if len(ds.Images) != 1 {
		t.Fatalf("got %d images, want 1 (non-image file should be skipped)", len(ds.Images))
	}
}
