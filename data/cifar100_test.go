package data

import (
	"os"
	"path/filepath"
	"testing"
)

// writeFakeCIFAR100Record appends one 3074-byte CIFAR-100 record (coarse
// label, fine label, 3072 pixel bytes) to buf.
func writeFakeCIFAR100Record(buf []byte, coarse, fine byte, pixelValue byte) []byte {
	record := make([]byte, 3074)
	record[0] = coarse
	record[1] = fine
	for i := 2; i < 3074; i++ {
		record[i] = pixelValue
	}
	return append(buf, record...)
}

func TestLoadCIFAR100BinaryParsesLabelsAndPixels(t *testing.T) {
	var buf []byte
	buf = writeFakeCIFAR100Record(buf, 5, 42, 255)
	buf = writeFakeCIFAR100Record(buf, 11, 7, 0)

	path := filepath.Join(t.TempDir(), "test.bin")
	if err := os.WriteFile(path, buf, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	dataset, err := LoadCIFAR100Binary(path)
	if err != nil {
		t.Fatalf("LoadCIFAR100Binary: %v", err)
	}
	if len(dataset.Images) != 2 {
		t.Fatalf("len(Images) = %d, want 2", len(dataset.Images))
	}
	if len(dataset.FineLabels) != 2 || len(dataset.CoarseLabels) != 2 {
		t.Fatalf("len(FineLabels)=%d len(CoarseLabels)=%d, want 2 and 2", len(dataset.FineLabels), len(dataset.CoarseLabels))
	}

	// Record 0: coarse=5, fine=42, all pixels 255 -> normalized 1.0.
	if dataset.CoarseLabels[0][5] != 1 {
		t.Errorf("record 0 CoarseLabels[5] = %v, want 1", dataset.CoarseLabels[0][5])
	}
	if dataset.FineLabels[0][42] != 1 {
		t.Errorf("record 0 FineLabels[42] = %v, want 1", dataset.FineLabels[0][42])
	}
	if dataset.Images[0].Data[0][0][0] != 1 {
		t.Errorf("record 0 pixel [0][0][0] = %v, want 1 (255/255)", dataset.Images[0].Data[0][0][0])
	}
	if dataset.Images[0].Height != 32 || dataset.Images[0].Width != 32 || dataset.Images[0].Channels != 3 {
		t.Errorf("record 0 image shape = (%d,%d,%d), want (32,32,3)", dataset.Images[0].Height, dataset.Images[0].Width, dataset.Images[0].Channels)
	}

	// Record 1: coarse=11, fine=7, all pixels 0 -> normalized 0.0.
	if dataset.CoarseLabels[1][11] != 1 {
		t.Errorf("record 1 CoarseLabels[11] = %v, want 1", dataset.CoarseLabels[1][11])
	}
	if dataset.FineLabels[1][7] != 1 {
		t.Errorf("record 1 FineLabels[7] = %v, want 1", dataset.FineLabels[1][7])
	}
	if dataset.Images[1].Data[31][31][2] != 0 {
		t.Errorf("record 1 pixel [31][31][2] = %v, want 0", dataset.Images[1].Data[31][31][2])
	}
}

func TestLoadCIFAR100BinaryBatchConcatenates(t *testing.T) {
	var buf1, buf2 []byte
	buf1 = writeFakeCIFAR100Record(buf1, 0, 0, 10)
	buf2 = writeFakeCIFAR100Record(buf2, 1, 1, 20)
	buf2 = writeFakeCIFAR100Record(buf2, 2, 2, 30)

	dir := t.TempDir()
	path1 := filepath.Join(dir, "part1.bin")
	path2 := filepath.Join(dir, "part2.bin")
	if err := os.WriteFile(path1, buf1, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if err := os.WriteFile(path2, buf2, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	dataset, err := LoadCIFAR100BinaryBatch([]string{path1, path2})
	if err != nil {
		t.Fatalf("LoadCIFAR100BinaryBatch: %v", err)
	}
	if len(dataset.Images) != 3 {
		t.Fatalf("len(Images) = %d, want 3", len(dataset.Images))
	}
}

func TestLoadCIFAR100ClassNames(t *testing.T) {
	dir := t.TempDir()
	finePath := filepath.Join(dir, "fine_label_names.txt")
	coarsePath := filepath.Join(dir, "coarse_label_names.txt")
	if err := os.WriteFile(finePath, []byte("apple\naquarium_fish\nbaby\n"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if err := os.WriteFile(coarsePath, []byte("aquatic_mammals\nfish\n"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	fine, coarse, err := LoadCIFAR100ClassNames(finePath, coarsePath)
	if err != nil {
		t.Fatalf("LoadCIFAR100ClassNames: %v", err)
	}
	wantFine := []string{"apple", "aquarium_fish", "baby"}
	if len(fine) != len(wantFine) {
		t.Fatalf("len(fine) = %d, want %d", len(fine), len(wantFine))
	}
	for i := range wantFine {
		if fine[i] != wantFine[i] {
			t.Errorf("fine[%d] = %q, want %q", i, fine[i], wantFine[i])
		}
	}
	wantCoarse := []string{"aquatic_mammals", "fish"}
	if len(coarse) != len(wantCoarse) {
		t.Fatalf("len(coarse) = %d, want %d", len(coarse), len(wantCoarse))
	}
	for i := range wantCoarse {
		if coarse[i] != wantCoarse[i] {
			t.Errorf("coarse[%d] = %q, want %q", i, coarse[i], wantCoarse[i])
		}
	}
}

func TestLoadCIFAR100ClassNamesMissingFileReturnsError(t *testing.T) {
	if _, _, err := LoadCIFAR100ClassNames(filepath.Join(t.TempDir(), "missing.txt"), filepath.Join(t.TempDir(), "also-missing.txt")); err == nil {
		t.Fatal("expected error for missing class-name files, got nil")
	}
}
