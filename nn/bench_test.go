package nn

import "testing"

// Sizes mirror the examples/cifar10_cnn showcase model's three conv stages
// and its dense head, batch 32 — realistic proxies for the workloads the
// parallelization and GEMM-style rewrite in nn/conv.go, nn/linear.go, and
// nn/norm.go target. Run with: go test ./nn/ -bench . -benchmem -run ^$

func benchConv(b *testing.B, inC, outC, hw int) {
	rng := NewRNG(1)
	layer := Conv2DSame(rng, inC, outC, 3, HeInit())
	x := NewTensor([]int{32, hw, hw, inC})
	ctx := &Context{Mode: Train}
	out, err := layer.Forward(ctx, x)
	if err != nil {
		b.Fatal(err)
	}
	gradOut := NewTensor(out.Shape)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err = layer.Forward(ctx, x)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := layer.Backward(ctx, gradOut); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConv2DStage1_3to16at32(b *testing.B)  { benchConv(b, 3, 16, 32) }
func BenchmarkConv2DStage2_16to32at16(b *testing.B) { benchConv(b, 16, 32, 16) }
func BenchmarkConv2DStage3_32to64at8(b *testing.B)  { benchConv(b, 32, 64, 8) }

func BenchmarkLinearHead_1024to128(b *testing.B) {
	rng := NewRNG(1)
	layer := Linear(rng, 1024, 128, HeInit())
	x := NewTensor([]int{32, 1024})
	ctx := &Context{Mode: Train}
	out, err := layer.Forward(ctx, x)
	if err != nil {
		b.Fatal(err)
	}
	gradOut := NewTensor(out.Shape)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err = layer.Forward(ctx, x)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := layer.Backward(ctx, gradOut); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBatchNorm_32at32x32(b *testing.B) {
	bn := BatchNorm(16)
	x := NewTensor([]int{32, 32, 32, 16})
	ctx := &Context{Mode: Train}
	out, err := bn.Forward(ctx, x)
	if err != nil {
		b.Fatal(err)
	}
	gradOut := NewTensor(out.Shape)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err = bn.Forward(ctx, x)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := bn.Backward(ctx, gradOut); err != nil {
			b.Fatal(err)
		}
	}
}
