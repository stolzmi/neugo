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
