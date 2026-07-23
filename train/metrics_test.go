package train

import (
	"math"
	"testing"

	"github.com/stolzmi/neugo/nn"
)

func TestComputeMetricsBinaryHandComputed(t *testing.T) {
	// 4 samples: predictions [0.9, 0.2, 0.6, 0.1], labels [1, 0, 1, 1]
	// predicted classes (>=0.5): 1,0,1,0 -> correct: samples 0,1,2 (3/4), sample 3 wrong
	pred, _ := nn.NewTensorFromData([]float32{0.9, 0.2, 0.6, 0.1}, []int{4, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 1, 1}, []int{4, 1})
	m, err := computeMetrics(0.1, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	if m.Accuracy != 75 {
		t.Errorf("Accuracy = %v, want 75", m.Accuracy)
	}
	// tp=2 (0,2), fp=0, tn=1 (1), fn=1 (3)
	wantCM := [][]int{{1, 0}, {1, 2}}
	for i := range wantCM {
		for j := range wantCM[i] {
			if m.ConfusionMatrix[i][j] != wantCM[i][j] {
				t.Errorf("ConfusionMatrix[%d][%d] = %d, want %d", i, j, m.ConfusionMatrix[i][j], wantCM[i][j])
			}
		}
	}
}

func TestComputeMetricsMulticlassHandComputed(t *testing.T) {
	// 3 samples, 3 classes, one-hot targets and argmax predictions:
	// sample0: pred class 0, actual class 0 (correct)
	// sample1: pred class 1, actual class 2 (wrong)
	// sample2: pred class 2, actual class 2 (correct)
	pred, _ := nn.NewTensorFromData([]float32{
		0.8, 0.1, 0.1,
		0.2, 0.7, 0.1,
		0.1, 0.2, 0.7,
	}, []int{3, 3})
	target, _ := nn.NewTensorFromData([]float32{
		1, 0, 0,
		0, 0, 1,
		0, 0, 1,
	}, []int{3, 3})
	m, err := computeMetrics(0.2, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	wantAcc := float32(2) / 3 * 100
	if diff := m.Accuracy - wantAcc; diff > 1e-4 || diff < -1e-4 {
		t.Errorf("Accuracy = %v, want %v", m.Accuracy, wantAcc)
	}
	if m.ConfusionMatrix[2][1] != 1 {
		t.Errorf("ConfusionMatrix[2][1] = %d, want 1 (actual=2 predicted=1)", m.ConfusionMatrix[2][1])
	}
}

func TestROCAUCBinaryHandComputed(t *testing.T) {
	// Same data as TestComputeMetricsBinaryHandComputed. Sorted ascending:
	// 0.1(pos) 0.2(neg) 0.6(pos) 0.9(pos) -> ranks 1,2,3,4; positive rank
	// sum = 1+3+4 = 8; nPos=3, nNeg=1.
	// AUC = (8 - 3*4/2) / (3*1) = 2/3.
	pred, _ := nn.NewTensorFromData([]float32{0.9, 0.2, 0.6, 0.1}, []int{4, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 1, 1}, []int{4, 1})
	m, err := computeMetrics(0.1, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	want := float32(2.0 / 3.0)
	if diff := math.Abs(float64(m.ROCAUC - want)); diff > 1e-4 {
		t.Errorf("ROCAUC = %v, want %v", m.ROCAUC, want)
	}
}

func TestPRAUCBinaryHandComputed(t *testing.T) {
	// Same data, sorted descending: 0.9(pos) 0.6(pos) 0.2(neg) 0.1(pos).
	// nPos=3: precision/recall at each positive step ->
	// (1, 1/3), (1, 2/3), (3/4, 1) with recall deltas 1/3,1/3,1/3 ->
	// AP = 1*1/3 + 1*1/3 + 0.75*1/3 = 0.91666...
	pred, _ := nn.NewTensorFromData([]float32{0.9, 0.2, 0.6, 0.1}, []int{4, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 1, 1}, []int{4, 1})
	m, err := computeMetrics(0.1, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	want := float32(11.0 / 12.0)
	if diff := math.Abs(float64(m.PRAUC - want)); diff > 1e-3 {
		t.Errorf("PRAUC = %v, want %v", m.PRAUC, want)
	}
}

func TestROCAUCAllTiedScoresEqualsChance(t *testing.T) {
	// Every score identical -> every rank identical -> AUC collapses to
	// the chance level 0.5, regardless of how ties are broken internally.
	scores := []float32{0.5, 0.5, 0.5, 0.5}
	labels := []float32{0, 0, 1, 1}
	auc, ok := rocAUCBinary(scores, labels)
	if !ok {
		t.Fatal("rocAUCBinary returned ok=false, want true (both classes present)")
	}
	if diff := math.Abs(float64(auc - 0.5)); diff > 1e-5 {
		t.Errorf("AUC with all-tied scores = %v, want 0.5", auc)
	}
}

func TestROCAUCUndefinedWhenOneClassMissing(t *testing.T) {
	_, ok := rocAUCBinary([]float32{0.1, 0.2, 0.3}, []float32{0, 0, 0})
	if ok {
		t.Fatal("rocAUCBinary with no positive examples returned ok=true, want false (undefined)")
	}
}

func TestBinaryTop5AccuracyEqualsAccuracy(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.9, 0.2, 0.6, 0.1}, []int{4, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 1, 1}, []int{4, 1})
	m, err := computeMetrics(0.1, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	if m.Top5Accuracy != m.Accuracy {
		t.Errorf("Top5Accuracy = %v, want it to equal Accuracy (%v) in the binary case", m.Top5Accuracy, m.Accuracy)
	}
}

func TestPerplexityIsExpOfLoss(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.9, 0.2}, []int{2, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0}, []int{2, 1})
	m, err := computeMetrics(1.0, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	want := float32(math.Exp(1.0))
	if diff := math.Abs(float64(m.Perplexity - want)); diff > 1e-4 {
		t.Errorf("Perplexity = %v, want %v (exp(1.0))", m.Perplexity, want)
	}
}

func TestMulticlassROCAUCAndPRAUCPerfectSeparation(t *testing.T) {
	// 2 classes, 4 samples: class 0 (positive for samples 0,1) is
	// perfectly separable from class 1 (positive for samples 2,3) in both
	// columns -> both classes' AUC/AP should be exactly 1.0, so the macro
	// average must also be exactly 1.0.
	pred, _ := nn.NewTensorFromData([]float32{
		0.9, 0.1,
		0.8, 0.2,
		0.3, 0.7,
		0.1, 0.9,
	}, []int{4, 2})
	target, _ := nn.NewTensorFromData([]float32{
		1, 0,
		1, 0,
		0, 1,
		0, 1,
	}, []int{4, 2})
	m, err := computeMetrics(0.1, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	if diff := math.Abs(float64(m.ROCAUC - 1.0)); diff > 1e-5 {
		t.Errorf("ROCAUC = %v, want 1.0 (perfect separation both classes)", m.ROCAUC)
	}
	if diff := math.Abs(float64(m.PRAUC - 1.0)); diff > 1e-5 {
		t.Errorf("PRAUC = %v, want 1.0 (perfect separation both classes)", m.PRAUC)
	}
}

func TestMulticlassTop5AccuracyWithMoreThanFiveClasses(t *testing.T) {
	// 6 classes, so top-5 is a genuine (non-trivial) subset.
	pred, _ := nn.NewTensorFromData([]float32{
		0.9, 0.8, 0.7, 0.6, 0.5, 0.1, // true class 5 is the *lowest* score -> excluded from top5
		0.1, 0.2, 0.05, 0.9, 0.5, 0.3, // true class 3 is the highest score -> included in top5
	}, []int{2, 6})
	target, _ := nn.NewTensorFromData([]float32{
		0, 0, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0,
	}, []int{2, 6})
	m, err := computeMetrics(0.1, pred, target)
	if err != nil {
		t.Fatalf("computeMetrics: %v", err)
	}
	want := float32(50)
	if diff := math.Abs(float64(m.Top5Accuracy - want)); diff > 1e-4 {
		t.Errorf("Top5Accuracy = %v, want %v", m.Top5Accuracy, want)
	}
}

func TestTopKIndicesOrderedDescending(t *testing.T) {
	idx := topKIndices([]float32{0.1, 0.9, 0.3, 0.7}, 2)
	want := []int{1, 3}
	if len(idx) != len(want) {
		t.Fatalf("topKIndices returned %d indices, want %d", len(idx), len(want))
	}
	for i := range want {
		if idx[i] != want[i] {
			t.Errorf("idx[%d] = %d, want %d", i, idx[i], want[i])
		}
	}
}
