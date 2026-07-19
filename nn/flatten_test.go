package nn

import "testing"

func TestFlattenForwardPreservesOrder(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{1, 2, 2, 2})
	f := Flatten()
	y, err := f.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if y.Shape[0] != 1 || y.Shape[1] != 8 {
		t.Fatalf("Flatten shape = %v, want [1 8]", y.Shape)
	}
	for i := range x.Data {
		if y.Data[i] != x.Data[i] {
			t.Errorf("y.Data[%d] = %v, want %v", i, y.Data[i], x.Data[i])
		}
	}
}

func TestFlattenBackwardReshapes(t *testing.T) {
	x := NewTensor([]int{2, 2, 2, 3})
	f := Flatten()
	ctx := &Context{}
	if _, err := f.Forward(ctx, x); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gradOut := NewTensor([]int{2, 12})
	for i := range gradOut.Data {
		gradOut.Data[i] = float32(i)
	}
	gradIn, err := f.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if gradIn.Shape[0] != 2 || gradIn.Shape[1] != 2 || gradIn.Shape[2] != 2 || gradIn.Shape[3] != 3 {
		t.Fatalf("gradIn shape = %v, want [2 2 2 3]", gradIn.Shape)
	}
	for i := range gradOut.Data {
		if gradIn.Data[i] != gradOut.Data[i] {
			t.Errorf("gradIn.Data[%d] = %v, want %v", i, gradIn.Data[i], gradOut.Data[i])
		}
	}
}

func TestFlattenGradient(t *testing.T) {
	x := NewTensor([]int{2, 2, 2, 2})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.1
	}
	checkInputGradient(t, Flatten(), &Context{}, x)
}
