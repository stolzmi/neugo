package nn

import (
	"math"
	"testing"
)

func TestBatchNormNormalizesTrainBatch(t *testing.T) {
	bn := BatchNorm(2)
	// channel 0: [1,3,5,7] mean=4 var=5; channel 1: [10,10,10,10] mean=10 var=0
	x, _ := NewTensorFromData([]float32{1, 10, 3, 10, 5, 10, 7, 10}, []int{4, 2})
	y, err := bn.Forward(&Context{Mode: Train}, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// gamma=1, beta=0 initially, so y should equal xhat.
	var mean0 float32
	for i := 0; i < 4; i++ {
		mean0 += y.Data[i*2]
	}
	mean0 /= 4
	if math.Abs(float64(mean0)) > 1e-4 {
		t.Fatalf("channel 0 normalized mean = %v, want ~0", mean0)
	}
	for i := 0; i < 4; i++ {
		if math.Abs(float64(y.Data[i*2+1])) > 1e-2 {
			t.Fatalf("channel 1 (zero variance) normalized value = %v, want ~0", y.Data[i*2+1])
		}
	}
}

func TestBatchNormUsesRunningStatsInInference(t *testing.T) {
	bn := BatchNorm(1)
	x, _ := NewTensorFromData([]float32{1, 2, 3, 4}, []int{4, 1})
	if _, err := bn.Forward(&Context{Mode: Train}, x); err != nil {
		t.Fatalf("Forward (train): %v", err)
	}
	// A single-sample inference pass can't compute its own batch stats;
	// it must reuse the running stats recorded during the Train pass above.
	single, _ := NewTensorFromData([]float32{100}, []int{1, 1})
	y, err := bn.Forward(&Context{Mode: Inference}, single)
	if err != nil {
		t.Fatalf("Forward (inference): %v", err)
	}
	// With running mean far below 100 and small running variance, the
	// normalized output should be large and positive, not exactly 0.
	if y.Data[0] < 5 {
		t.Fatalf("inference output = %v, want a large positive value reflecting the running stats", y.Data[0])
	}
}

func TestBatchNormInputGradientDense(t *testing.T) {
	bn := BatchNorm(3)
	x := NewTensor([]int{5, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.3 - 1.0
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, bn, ctx, x)
}

func TestBatchNormInputGradientConv(t *testing.T) {
	bn := BatchNorm(2)
	x := NewTensor([]int{2, 3, 3, 2})
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.2 - 0.4
	}
	ctx := &Context{Mode: Train}
	checkInputGradient(t, bn, ctx, x)
}

func TestBatchNormParamGradients(t *testing.T) {
	bn := BatchNorm(3)
	x := NewTensor([]int{5, 3})
	for i := range x.Data {
		x.Data[i] = float32(i%7)*0.25 - 0.8
	}
	ctx := &Context{Mode: Train}
	forward := func() (*Tensor, error) { return bn.Forward(ctx, x) }
	backward := func(g *Tensor) (*Tensor, error) { return bn.Backward(ctx, g) }
	for _, p := range bn.Params() {
		checkParamGradient(t, forward, backward, p)
	}
}
