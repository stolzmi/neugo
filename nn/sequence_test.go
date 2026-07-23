package nn

import "testing"

func TestLastTimestepForward(t *testing.T) {
	x, _ := NewTensorFromData([]float32{
		1, 2, // t=0
		3, 4, // t=1
		5, 6, // t=2
	}, []int{1, 3, 2})
	out, err := LastTimestep().Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []float32{5, 6}
	for i := range want {
		if out.Data[i] != want[i] {
			t.Errorf("out.Data[%d] = %v, want %v", i, out.Data[i], want[i])
		}
	}
}

func TestLastTimestepGradient(t *testing.T) {
	x := NewTensor([]int{2, 4, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%7) * 0.1
	}
	checkInputGradient(t, LastTimestep(), &Context{}, x)
}

func TestLastTimestepRejectsNon3DInput(t *testing.T) {
	l := LastTimestep()
	if _, err := l.OutputShape([]int{2, 3}); err == nil {
		t.Fatal("expected error for non-3D input, got nil")
	}
}

func TestRNNThenLastTimestepComposesViaSequential(t *testing.T) {
	rng := NewRNG(5)
	model, err := Sequential([]int{2, 3, 4}, RNN(rng, 4, 6, XavierInit()), LastTimestep())
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	x := NewTensor([]int{2, 3, 4})
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.05
	}
	out, err := model.Forward(&Context{}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []int{2, 6}
	for i := range want {
		if out.Shape[i] != want[i] {
			t.Fatalf("output shape = %v, want %v", out.Shape, want)
		}
	}
}
