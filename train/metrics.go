package train

import (
	"fmt"
	"math"
	"sort"

	"github.com/stolzmi/neugo/nn"
)

// Metrics holds evaluation results, macro-averaged for multiclass.
type Metrics struct {
	Loss            float32
	Accuracy        float32
	Precision       float32
	Recall          float32
	F1Score         float32
	ConfusionMatrix [][]int

	// ROCAUC and PRAUC are macro-averaged one-vs-rest for multiclass (each
	// class scored as its own binary problem, averaged over classes that
	// have at least one positive and one negative example — same
	// skip-undefined-support convention as Precision/Recall above); 0 when
	// every class is degenerate (all-same-label), which can't happen with
	// more than one sample of each class in a real dataset but is possible
	// on tiny/synthetic ones.
	ROCAUC float32
	PRAUC  float32
	// Top5Accuracy is top-min(5,classes) accuracy: the fraction of samples
	// where the true class is among the 5 highest-probability predicted
	// classes. For the binary (classes==1, single sigmoid output) case
	// there are only 2 conceptual classes and no meaningful top-5 to take,
	// so it's set equal to Accuracy there.
	Top5Accuracy float32
	// Perplexity is exp(Loss) — only meaningful when the model was
	// trained with a natural-log cross-entropy loss; it's still populated
	// for other losses (e.g. MSE) but isn't a meaningful quantity there.
	Perplexity float32
}

func argmaxRow(row []float32) int {
	maxIdx := 0
	maxVal := row[0]
	for i := 1; i < len(row); i++ {
		if row[i] > maxVal {
			maxVal = row[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// topKIndices returns the indices of the k largest values in row, sorted
// descending by value — argmaxRow generalized to more than one index.
func topKIndices(row []float32, k int) []int {
	if k > len(row) {
		k = len(row)
	}
	idx := make([]int, len(row))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return row[idx[a]] > row[idx[b]] })
	return idx[:k]
}

// rocAUCBinary computes the ROC-AUC of a binary scoring problem via the
// Mann-Whitney U / rank-sum identity (avoids an explicit threshold sweep):
// AUC = (sum of positive-example ranks - nPos*(nPos+1)/2) / (nPos*nNeg),
// with tied scores getting the average of their tied ranks. Returns
// ok=false when either class is entirely absent (AUC undefined).
func rocAUCBinary(scores, labels []float32) (auc float32, ok bool) {
	n := len(scores)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return scores[order[a]] < scores[order[b]] })

	ranks := make([]float64, n)
	i := 0
	for i < n {
		j := i
		for j < n && scores[order[j]] == scores[order[i]] {
			j++
		}
		avgRank := float64(i+1+j) / 2 // 1-indexed ranks [i+1, j] averaged
		for k := i; k < j; k++ {
			ranks[order[k]] = avgRank
		}
		i = j
	}

	var sumRanksPos float64
	var nPos, nNeg int
	for i, label := range labels {
		if label > 0 {
			sumRanksPos += ranks[i]
			nPos++
		} else {
			nNeg++
		}
	}
	if nPos == 0 || nNeg == 0 {
		return 0, false
	}
	return float32((sumRanksPos - float64(nPos)*float64(nPos+1)/2) / (float64(nPos) * float64(nNeg))), true
}

// prAUCBinary computes average precision (area under the precision-recall
// curve) for a binary scoring problem: sort by score descending, then sum
// precision*(recall delta) at each example — the standard step-function
// AP estimator. Returns ok=false when either class is entirely absent.
func prAUCBinary(scores, labels []float32) (ap float32, ok bool) {
	n := len(scores)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })

	var nPos int
	for _, label := range labels {
		if label > 0 {
			nPos++
		}
	}
	if nPos == 0 || nPos == n {
		return 0, false
	}

	var tp, fp int
	var sum, prevRecall float64
	for _, idx := range order {
		if labels[idx] > 0 {
			tp++
		} else {
			fp++
		}
		precision := float64(tp) / float64(tp+fp)
		recall := float64(tp) / float64(nPos)
		sum += precision * (recall - prevRecall)
		prevRecall = recall
	}
	return float32(sum), true
}

func computeMetrics(loss float32, pred, target *nn.Tensor) (Metrics, error) {
	if err := sameShape(pred, target); err != nil {
		return Metrics{}, err
	}
	if len(pred.Shape) != 2 {
		return Metrics{}, fmt.Errorf("train: computeMetrics expects [batch, classes], got %v", pred.Shape)
	}
	batch, classes := pred.Shape[0], pred.Shape[1]
	correct := 0

	if classes == 1 {
		var tp, fp, tn, fn int
		for b := 0; b < batch; b++ {
			predictedClass := 0
			if pred.Data[b] >= 0.5 {
				predictedClass = 1
			}
			actualClass := 0
			if target.Data[b] >= 0.5 {
				actualClass = 1
			}
			if predictedClass == actualClass {
				correct++
			}
			switch {
			case actualClass == 1 && predictedClass == 1:
				tp++
			case actualClass == 0 && predictedClass == 1:
				fp++
			case actualClass == 0 && predictedClass == 0:
				tn++
			case actualClass == 1 && predictedClass == 0:
				fn++
			}
		}
		var precision, recall, f1 float32
		if tp+fp > 0 {
			precision = float32(tp) / float32(tp+fp)
		}
		if tp+fn > 0 {
			recall = float32(tp) / float32(tp+fn)
		}
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}
		accuracy := float32(correct) / float32(batch) * 100
		rocAUC, _ := rocAUCBinary(pred.Data, target.Data)
		prAUC, _ := prAUCBinary(pred.Data, target.Data)
		return Metrics{
			Loss:            loss,
			Accuracy:        accuracy,
			Precision:       precision,
			Recall:          recall,
			F1Score:         f1,
			ConfusionMatrix: [][]int{{tn, fp}, {fn, tp}},
			ROCAUC:          rocAUC,
			PRAUC:           prAUC,
			Top5Accuracy:    accuracy,
			Perplexity:      float32(math.Exp(float64(loss))),
		}, nil
	}

	confusion := make([][]int, classes)
	for i := range confusion {
		confusion[i] = make([]int, classes)
	}
	top5K := classes
	if top5K > 5 {
		top5K = 5
	}
	top5Correct := 0
	for b := 0; b < batch; b++ {
		row := pred.Data[b*classes : (b+1)*classes]
		predClass := argmaxRow(row)
		actualClass := argmaxRow(target.Data[b*classes : (b+1)*classes])
		if predClass == actualClass {
			correct++
		}
		confusion[actualClass][predClass]++
		for _, idx := range topKIndices(row, top5K) {
			if idx == actualClass {
				top5Correct++
				break
			}
		}
	}

	var totalPrecision, totalRecall float32
	numClasses := 0
	var totalROCAUC, totalPRAUC float32
	numAUCClasses := 0
	classScores := make([]float32, batch)
	classLabels := make([]float32, batch)
	for c := 0; c < classes; c++ {
		tp := confusion[c][c]
		var fp, fn int
		for i := 0; i < classes; i++ {
			if i != c {
				fp += confusion[i][c]
				fn += confusion[c][i]
			}
		}
		if tp+fp > 0 {
			totalPrecision += float32(tp) / float32(tp+fp)
			numClasses++
		}
		if tp+fn > 0 {
			totalRecall += float32(tp) / float32(tp+fn)
		}

		for b := 0; b < batch; b++ {
			classScores[b] = pred.Data[b*classes+c]
			classLabels[b] = target.Data[b*classes+c]
		}
		if rocAUC, ok := rocAUCBinary(classScores, classLabels); ok {
			totalROCAUC += rocAUC
			if prAUC, ok := prAUCBinary(classScores, classLabels); ok {
				totalPRAUC += prAUC
			}
			numAUCClasses++
		}
	}
	var precision, recall, f1 float32
	if numClasses > 0 {
		precision = totalPrecision / float32(numClasses)
		recall = totalRecall / float32(numClasses)
	}
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}
	var rocAUC, prAUC float32
	if numAUCClasses > 0 {
		rocAUC = totalROCAUC / float32(numAUCClasses)
		prAUC = totalPRAUC / float32(numAUCClasses)
	}
	return Metrics{
		Loss:            loss,
		Accuracy:        float32(correct) / float32(batch) * 100,
		Precision:       precision,
		Recall:          recall,
		F1Score:         f1,
		ConfusionMatrix: confusion,
		ROCAUC:          rocAUC,
		PRAUC:           prAUC,
		Top5Accuracy:    float32(top5Correct) / float32(batch) * 100,
		Perplexity:      float32(math.Exp(float64(loss))),
	}, nil
}
