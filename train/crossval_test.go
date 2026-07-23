package train

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stolzmi/neugo/nn"
)

func newTestRNG(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

func makeXY(n int) ([][]float32, [][]float32) {
	x := make([][]float32, n)
	y := make([][]float32, n)
	for i := 0; i < n; i++ {
		x[i] = []float32{float32(i)}
		class := float32(0)
		if i%2 == 0 {
			class = 1
		}
		y[i] = []float32{class}
	}
	return x, y
}

func TestKFoldSplitsSizesAndNoOverlap(t *testing.T) {
	x, y := makeXY(10)
	folds := KFoldSplits(newTestRNG(1), x, y, 5, false)
	if len(folds) != 5 {
		t.Fatalf("len(folds) = %d, want 5", len(folds))
	}
	seenAsTest := map[float32]int{}
	for _, f := range folds {
		if len(f.TestX)+len(f.TrainX) != 10 {
			t.Fatalf("fold train+test = %d, want 10", len(f.TestX)+len(f.TrainX))
		}
		for _, row := range f.TestX {
			seenAsTest[row[0]]++
		}
	}
	if len(seenAsTest) != 10 {
		t.Fatalf("distinct samples seen as test = %d, want 10", len(seenAsTest))
	}
	for v, count := range seenAsTest {
		if count != 1 {
			t.Errorf("sample %v appeared as test in %d folds, want exactly 1", v, count)
		}
	}
}

func TestStratifiedKFoldPreservesClassRatioPerFold(t *testing.T) {
	x, y := makeXY(20) // exactly 10 class-1, 10 class-0
	folds := StratifiedKFoldSplits(newTestRNG(2), x, y, 4)
	for i, f := range folds {
		var ones int
		for _, row := range f.TestY {
			if row[0] > 0.5 {
				ones++
			}
		}
		if ones != len(f.TestY)/2 {
			t.Errorf("fold %d: %d/%d test samples are class 1, want exactly half", i, ones, len(f.TestY))
		}
	}
}

func TestCrossValidateAggregatesMeanAndStd(t *testing.T) {
	// Since folds now run concurrently, trainFold must derive its result
	// from the fold it's given rather than a shared counter incremented
	// in call order (call order across goroutines is not guaranteed to
	// match fold index order) — the target accuracy is encoded in each
	// fold's TrainX instead.
	accuracies := []float32{60, 70, 80}
	folds := make([]Fold, len(accuracies))
	for i, acc := range accuracies {
		folds[i] = Fold{TrainX: [][]float32{{acc}}}
	}
	trainFold := func(f Fold) (Metrics, error) {
		acc := f.TrainX[0][0]
		return Metrics{Accuracy: acc, F1Score: acc / 100, Loss: 1 - acc/100}, nil
	}
	result, err := CrossValidate(folds, trainFold)
	if err != nil {
		t.Fatalf("CrossValidate: %v", err)
	}
	if result.MeanAccuracy != 70 {
		t.Fatalf("MeanAccuracy = %v, want 70", result.MeanAccuracy)
	}
	if result.BestFold != 2 || result.WorstFold != 0 {
		t.Fatalf("BestFold=%d WorstFold=%d, want 2 and 0", result.BestFold, result.WorstFold)
	}
}

func TestCrossValidateRunsFoldsConcurrently(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("GOMAXPROCS < 2; cannot observe concurrency")
	}
	numFolds := runtime.GOMAXPROCS(0) * 2
	folds := make([]Fold, numFolds)

	var active, maxActive int32
	trainFold := func(f Fold) (Metrics, error) {
		n := atomic.AddInt32(&active, 1)
		for {
			cur := atomic.LoadInt32(&maxActive)
			if n <= cur || atomic.CompareAndSwapInt32(&maxActive, cur, n) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		atomic.AddInt32(&active, -1)
		return Metrics{}, nil
	}
	if _, err := CrossValidate(folds, trainFold); err != nil {
		t.Fatalf("CrossValidate: %v", err)
	}
	if atomic.LoadInt32(&maxActive) < 2 {
		t.Fatalf("max concurrent trainFold calls = %d, want >= 2 (folds should run in parallel)", maxActive)
	}
}

func TestCrossValidateDeterministicForcesSequentialFoldOrder(t *testing.T) {
	nn.SetDeterministic(true)
	defer nn.SetDeterministic(false)

	numFolds := 8
	folds := make([]Fold, numFolds)
	for i := range folds {
		folds[i] = Fold{TrainX: [][]float32{{float32(i)}}}
	}
	var order []int
	trainFold := func(f Fold) (Metrics, error) {
		order = append(order, int(f.TrainX[0][0]))
		return Metrics{}, nil
	}
	if _, err := CrossValidate(folds, trainFold); err != nil {
		t.Fatalf("CrossValidate: %v", err)
	}
	for i, v := range order {
		if v != i {
			t.Fatalf("order[%d] = %d, want %d (SetDeterministic(true) should force sequential fold order)", i, v, i)
		}
	}
}

func TestCrossValidatePropagatesFoldError(t *testing.T) {
	folds := []Fold{{}}
	_, err := CrossValidate(folds, func(f Fold) (Metrics, error) {
		return Metrics{}, fmt.Errorf("boom")
	})
	if err == nil {
		t.Fatal("expected error to propagate from a failing fold, got nil")
	}
}
