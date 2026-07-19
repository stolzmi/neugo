package nn

import "testing"

func TestNewTensorZeroed(t *testing.T) {
	tn := NewTensor([]int{2, 3})
	if tn.Size() != 6 {
		t.Fatalf("Size() = %d, want 6", tn.Size())
	}
	if len(tn.Data) != 6 {
		t.Fatalf("len(Data) = %d, want 6", len(tn.Data))
	}
	for i, v := range tn.Data {
		if v != 0 {
			t.Errorf("Data[%d] = %v, want 0", i, v)
		}
	}
}

func TestNewTensorFromDataShapeMismatch(t *testing.T) {
	_, err := NewTensorFromData([]float32{1, 2, 3}, []int{2, 2})
	if err == nil {
		t.Fatal("expected error for mismatched shape/data length, got nil")
	}
}

func TestTensorClone(t *testing.T) {
	orig, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	clone := orig.Clone()
	clone.Data[0] = 99
	if orig.Data[0] == 99 {
		t.Fatal("Clone shares underlying data with original")
	}
	if clone.Shape[0] != 2 || clone.Shape[1] != 2 {
		t.Fatalf("Clone shape = %v, want [2 2]", clone.Shape)
	}
}
