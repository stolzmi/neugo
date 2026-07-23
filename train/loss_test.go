package train

import (
	"math"
	"github.com/stolzmi/neugo/nn"
	"testing"
)

func TestMSELossValueAndGradient(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{1, 2, 3, 4}, []int{2, 2})
	target, _ := nn.NewTensorFromData([]float32{1, 1, 3, 6}, []int{2, 2})
	loss, grad, err := MSELoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// (0^2 + 1^2 + 0^2 + 2^2) / 4 = 5/4
	if diff := math.Abs(float64(loss - 1.25)); diff > 1e-5 {
		t.Fatalf("loss = %v, want 1.25", loss)
	}
	// d/dp mean((p-t)^2) = 2*(p-t)/N
	want := []float32{0, 0.5, 0, -1}
	for i := range want {
		if diff := math.Abs(float64(grad.Data[i] - want[i])); diff > 1e-5 {
			t.Errorf("grad[%d] = %v, want %v", i, grad.Data[i], want[i])
		}
	}
}

func TestBCELossClipsAndMatchesFiniteDifference(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.9, 0.1}, []int{2, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0}, []int{2, 1})
	_, grad, err := BCELoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	const eps = 1e-3
	for i := range pred.Data {
		p2 := pred.Clone()
		p2.Data[i] += eps
		lp, _, _ := BCELoss().Loss(p2, target)
		p3 := pred.Clone()
		p3.Data[i] -= eps
		lm, _, _ := BCELoss().Loss(p3, target)
		numGrad := (lp - lm) / (2 * eps)
		if diff := math.Abs(float64(numGrad - grad.Data[i])); diff > 1e-2 {
			t.Errorf("grad[%d] = %v, numeric = %v", i, grad.Data[i], numGrad)
		}
	}
}

func TestBCELossBatchSizeNormalization(t *testing.T) {
	// Test that BCE normalizes by batch size, not total element count.
	// With shape [2, 3] (2 batch samples, 3 features):
	// - Total elements = 6
	// - Batch size = 2
	// If bug normalizes by total elements instead of batch, values will be wrong.
	pred, _ := nn.NewTensorFromData([]float32{0.7, 0.8, 0.6, 0.9, 0.1, 0.5}, []int{2, 3})
	target, _ := nn.NewTensorFromData([]float32{1, 1, 0, 1, 0, 1}, []int{2, 3})
	loss, grad, err := BCELoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}

	// Expected loss (normalized by batch size 2, not total elements 6):
	// BCE = -mean_batch(y*log(p) + (1-y)*log(1-p))
	// Sum all 6 elements:
	// Element 0: y=1, p=0.7: -log(0.7) ≈ 0.3567
	// Element 1: y=1, p=0.8: -log(0.8) ≈ 0.2231
	// Element 2: y=0, p=0.6: -log(0.4) ≈ 0.9163
	// Element 3: y=1, p=0.9: -log(0.9) ≈ 0.1054
	// Element 4: y=0, p=0.1: -log(0.9) ≈ 0.1054
	// Element 5: y=1, p=0.5: -log(0.5) ≈ 0.6931
	// Sum ≈ 2.4000; divided by batch (2) = 1.2000
	wantLoss := float32(1.2)
	if diff := math.Abs(float64(loss - wantLoss)); diff > 1e-3 {
		t.Errorf("loss = %v, want ~%v", loss, wantLoss)
	}

	// Expected gradient: (-(y/p) + (1-y)/(1-p)) / batchSize
	// Each gradient element normalized by batch size (2), not total elements (6)
	// Row 0, Col 0: y=1, p=0.7: (-(1/0.7) + 0) / 2 ≈ -0.7143
	// Row 0, Col 1: y=1, p=0.8: (-(1/0.8) + 0) / 2 ≈ -0.625
	// Row 0, Col 2: y=0, p=0.6: (0 + (1/0.4)) / 2 = 1.25
	// Row 1, Col 0: y=1, p=0.9: (-(1/0.9) + 0) / 2 ≈ -0.5556
	// Row 1, Col 1: y=0, p=0.1: (0 + (1/0.9)) / 2 ≈ 0.5556
	// Row 1, Col 2: y=1, p=0.5: (-(1/0.5) + 0) / 2 = -1.0
	wantGrad := []float32{-0.7143, -0.625, 1.25, -0.5556, 0.5556, -1.0}
	for i := range wantGrad {
		if diff := math.Abs(float64(grad.Data[i] - wantGrad[i])); diff > 1e-3 {
			t.Errorf("grad[%d] = %v, want ~%v", i, grad.Data[i], wantGrad[i])
		}
	}
}

func TestCrossEntropyNonFusedComposesSoftmaxJacobian(t *testing.T) {
	logits, _ := nn.NewTensorFromData([]float32{2, 1, 0.1}, []int{1, 3})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 0}, []int{1, 3})
	ce := CrossEntropy() // fused defaults to false
	_, grad, err := ce.Loss(logits, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// gradient must equal softmax(logits)-target, composed internally
	sm := nn.Softmax()
	probs, _ := sm.Forward(&nn.Context{}, logits)
	for i := range probs.Data {
		want := probs.Data[i] - target.Data[i]
		if diff := math.Abs(float64(grad.Data[i] - want)); diff > 1e-5 {
			t.Errorf("grad[%d] = %v, want %v", i, grad.Data[i], want)
		}
	}
}

func TestCrossEntropyFusedUsesPredDirectly(t *testing.T) {
	probs, _ := nn.NewTensorFromData([]float32{0.7, 0.2, 0.1}, []int{1, 3})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 0}, []int{1, 3})
	ce := CrossEntropy()
	ce.SetFused(true)
	_, grad, err := ce.Loss(probs, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	for i := range probs.Data {
		want := probs.Data[i] - target.Data[i]
		if diff := math.Abs(float64(grad.Data[i] - want)); diff > 1e-5 {
			t.Errorf("grad[%d] = %v, want %v", i, grad.Data[i], want)
		}
	}
}

// checkLossGradient verifies loss.Loss's returned gradient against central
// finite differences of the scalar loss value, perturbing one element of
// pred at a time — the loss-level analogue of nn's checkInputGradient.
func checkLossGradient(t *testing.T, loss Loss, pred, target *nn.Tensor) {
	t.Helper()
	_, grad, err := loss.Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	const eps = 1e-3
	for i := range pred.Data {
		orig := pred.Data[i]
		pred.Data[i] = orig + eps
		lp, _, err := loss.Loss(pred, target)
		if err != nil {
			t.Fatalf("Loss(+eps): %v", err)
		}
		pred.Data[i] = orig - eps
		lm, _, err := loss.Loss(pred, target)
		if err != nil {
			t.Fatalf("Loss(-eps): %v", err)
		}
		pred.Data[i] = orig
		numGrad := (lp - lm) / (2 * eps)
		if diff := math.Abs(float64(numGrad - grad.Data[i])); diff > 1e-2 {
			t.Errorf("grad[%d] = %v, numeric = %v", i, grad.Data[i], numGrad)
		}
	}
}

func TestHuberLossQuadraticRegionMatchesMSEHalf(t *testing.T) {
	// |diff| <= delta everywhere -> Huber should equal 0.5*MSE elementwise.
	pred, _ := nn.NewTensorFromData([]float32{1, 2}, []int{1, 2})
	target, _ := nn.NewTensorFromData([]float32{1.2, 1.7}, []int{1, 2})
	loss, _, err := HuberLoss(1.0).Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	want := float32((0.5*0.2*0.2 + 0.5*0.3*0.3) / 2)
	if diff := math.Abs(float64(loss - want)); diff > 1e-5 {
		t.Fatalf("loss = %v, want %v", loss, want)
	}
}

func TestHuberLossLinearRegionBeyondDelta(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{5}, []int{1, 1})
	target, _ := nn.NewTensorFromData([]float32{0}, []int{1, 1})
	loss, grad, err := HuberLoss(1.0).Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// diff=5, delta=1: loss = delta*(|diff|-0.5*delta) = 1*(5-0.5) = 4.5
	if diff := math.Abs(float64(loss - 4.5)); diff > 1e-5 {
		t.Fatalf("loss = %v, want 4.5", loss)
	}
	if diff := math.Abs(float64(grad.Data[0] - 1.0)); diff > 1e-5 {
		t.Fatalf("grad[0] = %v, want 1.0 (delta, since n=1)", grad.Data[0])
	}
}

func TestHuberLossGradient(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.5, 2.5, -1.5, 0.1}, []int{1, 4})
	target, _ := nn.NewTensorFromData([]float32{0.6, 0.3, 0.2, 0.1}, []int{1, 4})
	checkLossGradient(t, HuberLoss(1.0), pred, target)
}

func TestSmoothL1LossIsHuberWithDeltaOne(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{3, -2}, []int{1, 2})
	target, _ := nn.NewTensorFromData([]float32{0, 0}, []int{1, 2})
	want, _, err := HuberLoss(1.0).Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	got, _, err := SmoothL1Loss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if got != want {
		t.Fatalf("SmoothL1Loss() = %v, want HuberLoss(1.0) = %v", got, want)
	}
}

func TestKLDivergenceZeroWhenDistributionsMatch(t *testing.T) {
	p, _ := nn.NewTensorFromData([]float32{0.2, 0.3, 0.5}, []int{1, 3})
	loss, _, err := KLDivergenceLoss().Loss(p, p)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if math.Abs(float64(loss)) > 1e-5 {
		t.Fatalf("KL(p||p) = %v, want 0", loss)
	}
}

func TestKLDivergenceGradient(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.6, 0.3, 0.1}, []int{1, 3})
	target, _ := nn.NewTensorFromData([]float32{0.2, 0.5, 0.3}, []int{1, 3})
	checkLossGradient(t, KLDivergenceLoss(), pred, target)
}

func TestHingeLossZeroWhenMarginSatisfied(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{2, -2}, []int{1, 2})
	target, _ := nn.NewTensorFromData([]float32{1, -1}, []int{1, 2})
	loss, grad, err := HingeLoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if loss != 0 {
		t.Fatalf("loss = %v, want 0 (margin satisfied)", loss)
	}
	for i, g := range grad.Data {
		if g != 0 {
			t.Errorf("grad[%d] = %v, want 0", i, g)
		}
	}
}

func TestHingeLossPositiveWhenMarginViolated(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.2}, []int{1, 1})
	target, _ := nn.NewTensorFromData([]float32{1}, []int{1, 1})
	loss, grad, err := HingeLoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// margin = 1 - 1*0.2 = 0.8
	if diff := math.Abs(float64(loss - 0.8)); diff > 1e-5 {
		t.Fatalf("loss = %v, want 0.8", loss)
	}
	if diff := math.Abs(float64(grad.Data[0] - -1.0)); diff > 1e-5 {
		t.Fatalf("grad[0] = %v, want -1.0 (n=1)", grad.Data[0])
	}
}

func TestFocalLossGammaZeroAlphaHalfMatchesScaledBCE(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.7, 0.3}, []int{2, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0}, []int{2, 1})
	focalLoss, _, err := FocalLoss(0, 0.5).Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	bceLoss, _, err := BCELoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// Gamma=0 removes the focusing term; Alpha=0.5 on both classes reduces
	// to 0.5*BCE.
	if diff := math.Abs(float64(focalLoss - 0.5*bceLoss)); diff > 1e-4 {
		t.Fatalf("FocalLoss(0, 0.5) = %v, want 0.5*BCE = %v", focalLoss, 0.5*bceLoss)
	}
}

func TestFocalLossGradient(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.8, 0.3, 0.6, 0.1}, []int{4, 1})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 1, 0}, []int{4, 1})
	checkLossGradient(t, FocalLoss(2.0, 0.25), pred, target)
}

func TestCosineSimilarityZeroWhenVectorsAligned(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{3, 4}, []int{1, 2})
	target, _ := nn.NewTensorFromData([]float32{6, 8}, []int{1, 2}) // same direction, different magnitude
	loss, _, err := CosineSimilarityLoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if math.Abs(float64(loss)) > 1e-4 {
		t.Fatalf("loss = %v, want ~0 for aligned vectors", loss)
	}
}

func TestCosineSimilarityMaxWhenVectorsOpposed(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{1, 0}, []int{1, 2})
	target, _ := nn.NewTensorFromData([]float32{-1, 0}, []int{1, 2})
	loss, _, err := CosineSimilarityLoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if diff := math.Abs(float64(loss - 2.0)); diff > 1e-4 {
		t.Fatalf("loss = %v, want 2.0 for opposed vectors", loss)
	}
}

func TestCosineSimilarityGradient(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{0.6, -0.4, 0.9, 1.2, 0.3, -0.7}, []int{2, 3})
	target, _ := nn.NewTensorFromData([]float32{0.2, 0.5, -0.3, -0.1, 0.8, 0.4}, []int{2, 3})
	checkLossGradient(t, CosineSimilarityLoss(), pred, target)
}

func TestLabelSmoothingPreservesProbabilityMass(t *testing.T) {
	logits, _ := nn.NewTensorFromData([]float32{2, 1, 0.1}, []int{1, 3})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 0}, []int{1, 3})
	smoothed := LabelSmoothingLoss(CrossEntropy(), 0.1, 3)
	lossSmoothed, _, err := smoothed.Loss(logits, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	lossPlain, _, err := CrossEntropy().Loss(logits, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	// Smoothing spreads probability mass away from the true class, which
	// can only increase cross-entropy against a one-hot target.
	if lossSmoothed <= lossPlain {
		t.Fatalf("smoothed loss %v should exceed plain CE loss %v", lossSmoothed, lossPlain)
	}
}

func TestLabelSmoothingZeroEpsilonMatchesInner(t *testing.T) {
	logits, _ := nn.NewTensorFromData([]float32{2, 1, 0.1}, []int{1, 3})
	target, _ := nn.NewTensorFromData([]float32{1, 0, 0}, []int{1, 3})
	smoothed := LabelSmoothingLoss(CrossEntropy(), 0, 3)
	lossSmoothed, gradSmoothed, err := smoothed.Loss(logits, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	lossPlain, gradPlain, err := CrossEntropy().Loss(logits, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if diff := math.Abs(float64(lossSmoothed - lossPlain)); diff > 1e-6 {
		t.Fatalf("Epsilon=0 loss = %v, want %v (plain CE)", lossSmoothed, lossPlain)
	}
	for i := range gradPlain.Data {
		if diff := math.Abs(float64(gradSmoothed.Data[i] - gradPlain.Data[i])); diff > 1e-6 {
			t.Errorf("grad[%d] = %v, want %v", i, gradSmoothed.Data[i], gradPlain.Data[i])
		}
	}
}

func TestMAELossGradientSign(t *testing.T) {
	pred, _ := nn.NewTensorFromData([]float32{5, 1}, []int{2, 1})
	target, _ := nn.NewTensorFromData([]float32{3, 4}, []int{2, 1})
	_, grad, err := MAELoss().Loss(pred, target)
	if err != nil {
		t.Fatalf("Loss: %v", err)
	}
	if grad.Data[0] <= 0 {
		t.Errorf("grad[0] = %v, want > 0 (pred > target)", grad.Data[0])
	}
	if grad.Data[1] >= 0 {
		t.Errorf("grad[1] = %v, want < 0 (pred < target)", grad.Data[1])
	}
}
