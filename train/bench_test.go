package train

import (
	"testing"

	"github.com/stolzmi/neugo/nn"
)

// Element counts mirror realistic parameter sizes: a 1024x1024 dense
// weight matrix (~1M elements, typical for a transformer FFN layer) and a
// 50000x256 embedding table (~12.8M elements, typical vocab x d_model).
// Run with: go test ./train/ -bench BenchmarkOptimizerStep -benchmem -run ^$
//
// Each parallel benchmark has a "_Sequential" twin that forces
// nn.SetDeterministic(true) (which this package's maxParallelChunks also
// honors) so the parallel-vs-single-threaded difference shows up directly
// in one benchmark run, without needing a separate GOMAXPROCS=1 process.
func benchOptimizerStep(b *testing.B, newOpt func() Optimizer, n int) {
	value := make([]float32, n)
	grad := make([]float32, n)
	for i := range value {
		value[i] = float32(i%13) * 0.01
		grad[i] = float32((i%9)-4) * 0.1
	}
	p := newTestParam(value, grad)
	opt := newOpt()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		opt.Step([]*nn.Param{p})
	}
}

func BenchmarkAdamStep_1M(b *testing.B) {
	benchOptimizerStep(b, func() Optimizer { return Adam(0.001, 0.9, 0.999, 1e-8) }, 1<<20)
}

func BenchmarkAdamStep_1M_Sequential(b *testing.B) {
	nn.SetDeterministic(true)
	defer nn.SetDeterministic(false)
	benchOptimizerStep(b, func() Optimizer { return Adam(0.001, 0.9, 0.999, 1e-8) }, 1<<20)
}

func BenchmarkAdamStep_Embedding12M(b *testing.B) {
	benchOptimizerStep(b, func() Optimizer { return Adam(0.001, 0.9, 0.999, 1e-8) }, 50000*256)
}

func BenchmarkAdamStep_Embedding12M_Sequential(b *testing.B) {
	nn.SetDeterministic(true)
	defer nn.SetDeterministic(false)
	benchOptimizerStep(b, func() Optimizer { return Adam(0.001, 0.9, 0.999, 1e-8) }, 50000*256)
}

func BenchmarkSGDStep_1M(b *testing.B) {
	benchOptimizerStep(b, func() Optimizer { return SGD(0.01) }, 1<<20)
}

func BenchmarkSGDStep_1M_Sequential(b *testing.B) {
	nn.SetDeterministic(true)
	defer nn.SetDeterministic(false)
	benchOptimizerStep(b, func() Optimizer { return SGD(0.01) }, 1<<20)
}

// benchCrossValidate simulates a CPU-bound trainFold (a plain arithmetic
// loop rather than an actual model, so the benchmark stays fast and has
// no other dependencies) to demonstrate CrossValidate's fold-level
// parallelism independent of any specific model's cost.
func benchCrossValidate(b *testing.B, numFolds int) {
	folds := make([]Fold, numFolds)
	trainFold := func(f Fold) (Metrics, error) {
		sum := 0.0
		for i := 0; i < 5_000_000; i++ {
			sum += float64(i) * 1.0000001
		}
		return Metrics{Accuracy: float32(sum)}, nil
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := CrossValidate(folds, trainFold); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCrossValidate_8Folds(b *testing.B) {
	benchCrossValidate(b, 8)
}

func BenchmarkCrossValidate_8Folds_Sequential(b *testing.B) {
	nn.SetDeterministic(true)
	defer nn.SetDeterministic(false)
	benchCrossValidate(b, 8)
}
