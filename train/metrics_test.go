package train

import (
	"neugo/nn"
	"testing"
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
