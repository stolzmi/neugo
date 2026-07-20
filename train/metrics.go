package train

import (
	"fmt"
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
		return Metrics{
			Loss:            loss,
			Accuracy:        float32(correct) / float32(batch) * 100,
			Precision:       precision,
			Recall:          recall,
			F1Score:         f1,
			ConfusionMatrix: [][]int{{tn, fp}, {fn, tp}},
		}, nil
	}

	confusion := make([][]int, classes)
	for i := range confusion {
		confusion[i] = make([]int, classes)
	}
	for b := 0; b < batch; b++ {
		predClass := argmaxRow(pred.Data[b*classes : (b+1)*classes])
		actualClass := argmaxRow(target.Data[b*classes : (b+1)*classes])
		if predClass == actualClass {
			correct++
		}
		confusion[actualClass][predClass]++
	}

	var totalPrecision, totalRecall float32
	numClasses := 0
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
	}
	var precision, recall, f1 float32
	if numClasses > 0 {
		precision = totalPrecision / float32(numClasses)
		recall = totalRecall / float32(numClasses)
	}
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}
	return Metrics{
		Loss:            loss,
		Accuracy:        float32(correct) / float32(batch) * 100,
		Precision:       precision,
		Recall:          recall,
		F1Score:         f1,
		ConfusionMatrix: confusion,
	}, nil
}
