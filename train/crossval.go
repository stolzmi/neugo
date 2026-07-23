package train

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/stolzmi/neugo/nn"
)

type Fold struct {
	TrainX, TrainY, TestX, TestY [][]float32
}

func KFoldSplits(rng *rand.Rand, x, y [][]float32, k int, shuffle bool) []Fold {
	n := len(x)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	if shuffle {
		rng.Shuffle(n, func(i, j int) { order[i], order[j] = order[j], order[i] })
	}

	foldSize := n / k
	folds := make([]Fold, k)
	for f := 0; f < k; f++ {
		start := f * foldSize
		end := start + foldSize
		if f == k-1 {
			end = n
		}
		var fold Fold
		for i, idx := range order {
			if i >= start && i < end {
				fold.TestX = append(fold.TestX, x[idx])
				fold.TestY = append(fold.TestY, y[idx])
			} else {
				fold.TrainX = append(fold.TrainX, x[idx])
				fold.TrainY = append(fold.TrainY, y[idx])
			}
		}
		folds[f] = fold
	}
	return folds
}

// StratifiedKFoldSplits assumes binary labels (y[i][0] > 0.5 == class 1)
// and preserves the class ratio in every fold's test split.
func StratifiedKFoldSplits(rng *rand.Rand, x, y [][]float32, k int) []Fold {
	var x0, y0, x1, y1 [][]float32
	for i := range y {
		if y[i][0] > 0.5 {
			x1, y1 = append(x1, x[i]), append(y1, y[i])
		} else {
			x0, y0 = append(x0, x[i]), append(y0, y[i])
		}
	}
	folds0 := KFoldSplits(rng, x0, y0, k, true)
	folds1 := KFoldSplits(rng, x1, y1, k, true)

	folds := make([]Fold, k)
	for f := 0; f < k; f++ {
		folds[f] = Fold{
			TrainX: append(append([][]float32{}, folds0[f].TrainX...), folds1[f].TrainX...),
			TrainY: append(append([][]float32{}, folds0[f].TrainY...), folds1[f].TrainY...),
			TestX:  append(append([][]float32{}, folds0[f].TestX...), folds1[f].TestX...),
			TestY:  append(append([][]float32{}, folds0[f].TestY...), folds1[f].TestY...),
		}
	}
	return folds
}

type CrossValResult struct {
	FoldMetrics               []Metrics
	MeanAccuracy, StdAccuracy float32
	MeanF1, StdF1             float32
	MeanLoss, StdLoss         float32
	BestFold, WorstFold       int
}

// CrossValidate calls trainFold once per fold — trainFold is expected to
// build a fresh model, train it on fold.TrainX/TrainY, and return the
// Metrics from evaluating it on fold.TestX/TestY. Folds run concurrently
// across a worker pool bounded by GOMAXPROCS (mirroring tune.Run's
// pattern), so trainFold must be safe to call from multiple goroutines at
// once — in practice this just means it must build its own fresh model
// and optimizer per call rather than closing over shared mutable state.
// Each fold writes to its own disjoint result slot, so no locking is
// needed beyond the wait for all folds to finish. With
// nn.SetDeterministic(true) in effect, folds run sequentially instead (in
// fold order) for bit-exact reproducibility, matching that switch's
// meaning elsewhere in the library.
func CrossValidate(folds []Fold, trainFold func(fold Fold) (Metrics, error)) (CrossValResult, error) {
	foldMetrics := make([]Metrics, len(folds))
	errs := make([]error, len(folds))

	if nn.IsDeterministic() {
		for i, f := range folds {
			m, err := trainFold(f)
			errs[i] = err
			foldMetrics[i] = m
		}
	} else {
		workers := runtime.GOMAXPROCS(0)
		if workers > len(folds) {
			workers = len(folds)
		}
		if workers < 1 {
			workers = 1
		}

		foldChan := make(chan int, len(folds))
		for i := range folds {
			foldChan <- i
		}
		close(foldChan)

		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range foldChan {
					m, err := trainFold(folds[i])
					errs[i] = err
					foldMetrics[i] = m
				}
			}()
		}
		wg.Wait()
	}

	for i, err := range errs {
		if err != nil {
			return CrossValResult{}, fmt.Errorf("train: fold %d: %w", i, err)
		}
	}
	return summarizeFolds(foldMetrics), nil
}

func summarizeFolds(metrics []Metrics) CrossValResult {
	k := len(metrics)
	result := CrossValResult{FoldMetrics: metrics}
	bestAcc := float32(math.Inf(-1))
	worstAcc := float32(math.Inf(1))
	var sumAcc, sumF1, sumLoss float32
	for i, m := range metrics {
		sumAcc += m.Accuracy
		sumF1 += m.F1Score
		sumLoss += m.Loss
		if m.Accuracy > bestAcc {
			bestAcc = m.Accuracy
			result.BestFold = i
		}
		if m.Accuracy < worstAcc {
			worstAcc = m.Accuracy
			result.WorstFold = i
		}
	}
	result.MeanAccuracy = sumAcc / float32(k)
	result.MeanF1 = sumF1 / float32(k)
	result.MeanLoss = sumLoss / float32(k)

	var varAcc, varF1, varLoss float32
	for _, m := range metrics {
		varAcc += (m.Accuracy - result.MeanAccuracy) * (m.Accuracy - result.MeanAccuracy)
		varF1 += (m.F1Score - result.MeanF1) * (m.F1Score - result.MeanF1)
		varLoss += (m.Loss - result.MeanLoss) * (m.Loss - result.MeanLoss)
	}
	result.StdAccuracy = float32(math.Sqrt(float64(varAcc / float32(k))))
	result.StdF1 = float32(math.Sqrt(float64(varF1 / float32(k))))
	result.StdLoss = float32(math.Sqrt(float64(varLoss / float32(k))))
	return result
}
