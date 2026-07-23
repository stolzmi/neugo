package nn

import "testing"

func TestMaxPool2DForward(t *testing.T) {
	// 1x4x4x1, pool 2x2 stride 2 -> 1x2x2x1
	x, _ := NewTensorFromData([]float32{
		1, 2, 5, 6,
		3, 4, 7, 8,
		9, 10, 13, 14,
		11, 12, 15, 16,
	}, []int{1, 4, 4, 1})
	m := MaxPool2D(2, 2)
	y, err := m.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []float32{4, 8, 12, 16}
	for i := range want {
		if y.Data[i] != want[i] {
			t.Errorf("y.Data[%d] = %v, want %v", i, y.Data[i], want[i])
		}
	}
}

func TestMaxPool2DGradient(t *testing.T) {
	x := NewTensor([]int{1, 4, 4, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%11) * 0.13
	}
	checkInputGradient(t, MaxPool2D(2, 2), &Context{}, x)
}

func TestAvgPool2DForward(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{1, 2, 2, 1})
	a := AvgPool2D(2, 2)
	y, err := a.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if diff := y.Data[0] - 2.5; diff > 1e-5 || diff < -1e-5 {
		t.Fatalf("y.Data[0] = %v, want 2.5", y.Data[0])
	}
}

func TestAvgPool2DGradient(t *testing.T) {
	x := NewTensor([]int{1, 4, 4, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%9) * 0.11
	}
	checkInputGradient(t, AvgPool2D(2, 2), &Context{}, x)
}

func TestMaxPool2DZeroStrideReturnsErrorNotPanic(t *testing.T) {
	_, err := Sequential([]int{1, 4, 4, 1}, MaxPool2D(2, 0))
	if err == nil {
		t.Fatal("Sequential with MaxPool2D(2, 0) returned nil error, want a clean error instead of a panic")
	}
}

func TestAvgPool2DZeroStrideReturnsErrorNotPanic(t *testing.T) {
	_, err := Sequential([]int{1, 4, 4, 1}, AvgPool2D(2, 0))
	if err == nil {
		t.Fatal("Sequential with AvgPool2D(2, 0) returned nil error, want a clean error instead of a panic")
	}
}

func TestMaxPool2DZeroPoolSizeReturnsError(t *testing.T) {
	_, err := MaxPool2D(0, 2).OutputShape([]int{1, 4, 4, 1})
	if err == nil {
		t.Fatal("MaxPool2D(0, 2).OutputShape returned nil error, want a clean error")
	}
}

func TestAvgPool2DZeroPoolSizeReturnsError(t *testing.T) {
	_, err := AvgPool2D(0, 2).OutputShape([]int{1, 4, 4, 1})
	if err == nil {
		t.Fatal("AvgPool2D(0, 2).OutputShape returned nil error, want a clean error")
	}
}

func TestAdaptiveAvgPool2DForwardUnevenBins(t *testing.T) {
	// 1x3x3x1 -> 1x2x2x1: with inSize=3, outSize=2, adjacent bins share
	// their boundary row/column (PyTorch's documented adaptive-pooling
	// behavior — bins are [0,2) and [1,3) along each axis, not a clean
	// 2/1 split).
	x, _ := NewTensorFromData([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, []int{1, 3, 3, 1})
	a := AdaptiveAvgPool2D(2, 2)
	y, err := a.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// Bin (0,0): rows{0,1} x cols{0,1} -> {1,2,4,5} mean 3.0
	// Bin (0,1): rows{0,1} x cols{1,2} -> {2,3,5,6} mean 4.0
	// Bin (1,0): rows{1,2} x cols{0,1} -> {4,5,7,8} mean 6.0
	// Bin (1,1): rows{1,2} x cols{1,2} -> {5,6,8,9} mean 7.0
	want := []float32{3.0, 4.0, 6.0, 7.0}
	for i := range want {
		if diff := y.Data[i] - want[i]; diff > 1e-5 || diff < -1e-5 {
			t.Errorf("y.Data[%d] = %v, want %v", i, y.Data[i], want[i])
		}
	}
}

func TestGlobalAvgPool2DMatchesAdaptiveOneByOne(t *testing.T) {
	x := NewTensor([]int{2, 3, 3, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%11) * 0.17
	}
	global, err := GlobalAvgPool2D().Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	adaptive, err := AdaptiveAvgPool2D(1, 1).Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := range global.Data {
		if global.Data[i] != adaptive.Data[i] {
			t.Errorf("GlobalAvgPool2D()[%d] = %v, want %v (AdaptiveAvgPool2D(1,1))", i, global.Data[i], adaptive.Data[i])
		}
	}
}

func TestAdaptiveAvgPool2DGradient(t *testing.T) {
	x := NewTensor([]int{1, 5, 5, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%13) * 0.09
	}
	checkInputGradient(t, AdaptiveAvgPool2D(2, 3), &Context{}, x)
}

func TestAdaptiveAvgPool2DRejectsTargetLargerThanInput(t *testing.T) {
	_, err := AdaptiveAvgPool2D(5, 5).OutputShape([]int{1, 4, 4, 1})
	if err == nil {
		t.Fatal("expected error when target size exceeds input spatial size, got nil")
	}
}

func TestGlobalMaxPool2DForward(t *testing.T) {
	x, _ := NewTensorFromData([]float32{
		1, 2, 5, 6,
		3, 4, 7, 8,
		9, 10, 13, 14,
		11, 12, 15, 16,
	}, []int{1, 4, 4, 1})
	y, err := GlobalMaxPool2D().Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if y.Data[0] != 16 {
		t.Errorf("y.Data[0] = %v, want 16", y.Data[0])
	}
}

func TestGlobalMaxPool2DGradient(t *testing.T) {
	x := NewTensor([]int{2, 4, 4, 3})
	// Every element distinct (no modulo) so each channel's global max is
	// unambiguous — a tie would make the numeric/analytic gradient mismatch
	// at the boundary between two equal maxima.
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.031
	}
	checkInputGradient(t, GlobalMaxPool2D(), &Context{}, x)
}
