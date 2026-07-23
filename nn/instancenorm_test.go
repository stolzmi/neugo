package nn

import "testing"

func TestInstanceNormIndependentPerChannelAndSample(t *testing.T) {
	// InstanceNorm's stats must depend on neither other channels nor other
	// samples — unlike GroupNorm(1, channels), which pools across all
	// channels, and unlike BatchNorm, which pools across the batch.
	in := InstanceNorm(2)
	ctx := &Context{Mode: Train}
	x := NewTensor([]int{2, 2, 2, 2}) // [batch, h, w, channels]
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.3
	}
	out1, err := in.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	channel0Sample0 := append([]float32(nil), out1.Data[0], out1.Data[2], out1.Data[4], out1.Data[6])

	x2 := x.Clone()
	for i := 1; i < len(x2.Data); i += 2 {
		x2.Data[i] = 100 // wildly different second channel, every sample
	}
	out2, err := in.Forward(ctx, x2)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	got := []float32{out2.Data[0], out2.Data[2], out2.Data[4], out2.Data[6]}
	for i := range channel0Sample0 {
		if diff := out1Diff(channel0Sample0[i], got[i]); diff > 1e-4 {
			t.Errorf("channel 0 output[%d] changed from %v to %v when channel 1 changed", i, channel0Sample0[i], got[i])
		}
	}
}

func TestInstanceNormGradients(t *testing.T) {
	in := InstanceNorm(3)
	x := NewTensor([]int{2, 3, 3, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%13)*0.05 - 0.3
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, in, ctx, x)
	forward := func() (*Tensor, error) { return in.Forward(ctx, x) }
	backward := func(grad *Tensor) (*Tensor, error) { return in.Backward(ctx, grad) }
	for _, p := range in.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}
