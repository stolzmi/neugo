package nn

import "testing"

func TestDropoutIdentityInInferenceMode(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{4})
	d := Dropout(0.9) // even at a high rate, inference must be identity
	y, err := d.Forward(&Context{Mode: Inference}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := range x.Data {
		if y.Data[i] != x.Data[i] {
			t.Errorf("y.Data[%d] = %v, want %v (identity)", i, y.Data[i], x.Data[i])
		}
	}
}

func TestDropoutIdentityWhenRateZero(t *testing.T) {
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{4})
	d := Dropout(0)
	y, err := d.Forward(&Context{Mode: Train, RNG: NewRNG(1)}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	for i := range x.Data {
		if y.Data[i] != x.Data[i] {
			t.Errorf("y.Data[%d] = %v, want %v (identity at rate 0)", i, y.Data[i], x.Data[i])
		}
	}
}

func TestDropoutApproximatesRateStatistically(t *testing.T) {
	rate := float32(0.3)
	x := NewTensor([]int{10000})
	for i := range x.Data {
		x.Data[i] = 1
	}
	d := Dropout(rate)
	y, err := d.Forward(&Context{Mode: Train, RNG: NewRNG(9)}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	zeroed := 0
	for _, v := range y.Data {
		if v == 0 {
			zeroed++
		}
	}
	frac := float64(zeroed) / float64(len(y.Data))
	if frac < 0.25 || frac > 0.35 {
		t.Fatalf("dropped fraction = %v, want close to %v", frac, rate)
	}
}

func TestDropoutBackwardScalesByRecordedMask(t *testing.T) {
	d := Dropout(0.5)
	x := NewTensor([]int{20})
	for i := range x.Data {
		x.Data[i] = 1
	}
	ctx := &Context{Mode: Train, RNG: NewRNG(7)}
	y, err := d.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	gradOut := NewTensor([]int{20})
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	gradIn, err := d.Backward(ctx, gradOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	for i := range gradIn.Data {
		if y.Data[i] == 0 {
			if gradIn.Data[i] != 0 {
				t.Errorf("gradIn[%d] = %v, want 0 (element was dropped)", i, gradIn.Data[i])
			}
		} else if diff := gradIn.Data[i] - y.Data[i]; diff > 1e-5 || diff < -1e-5 {
			// x[i]==1, so y[i] IS the scale factor applied; gradIn should match it exactly.
			t.Errorf("gradIn[%d] = %v, want %v", i, gradIn.Data[i], y.Data[i])
		}
	}
}
