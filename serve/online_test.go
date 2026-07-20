package serve

import (
	"bytes"
	"context"
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

func xorModel(t *testing.T) *nn.SequentialModel {
	rng := rand.New(rand.NewSource(42))
	model, err := nn.Sequential([]int{2, 2},
		nn.Linear(rng, 2, 4, nil),
		nn.ReLU(),
		nn.Linear(rng, 4, 1, nil),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Failed to create XOR model: %v", err)
	}
	return model
}

func xorSamples() []Sample {
	return []Sample{
		{X: []float32{0, 0}, Y: []float32{0}},
		{X: []float32{0, 1}, Y: []float32{1}},
		{X: []float32{1, 0}, Y: []float32{1}},
		{X: []float32{1, 1}, Y: []float32{0}},
	}
}

func xorHoldout() []Sample {
	return []Sample{
		{X: []float32{0, 0}, Y: []float32{0}},
		{X: []float32{0, 1}, Y: []float32{1}},
		{X: []float32{1, 0}, Y: []float32{1}},
		{X: []float32{1, 1}, Y: []float32{0}},
	}
}

func TestRingBufferOverwritesOldest(t *testing.T) {
	rb := newRingBuffer(4)

	samples := []Sample{
		{X: []float32{1}, Y: []float32{1}},
		{X: []float32{2}, Y: []float32{2}},
		{X: []float32{3}, Y: []float32{3}},
		{X: []float32{4}, Y: []float32{4}},
		{X: []float32{5}, Y: []float32{5}},
		{X: []float32{6}, Y: []float32{6}},
	}

	for _, s := range samples {
		rb.Push(s)
	}

	snapshot := rb.Snapshot()
	if len(snapshot) != 4 {
		t.Fatalf("Expected 4 samples in snapshot, got %d", len(snapshot))
	}

	// Should have the last 4 samples: [3, 4, 5, 6]
	expected := []float32{3, 4, 5, 6}
	for i, s := range snapshot {
		if len(s.X) != 1 || s.X[0] != expected[i] {
			t.Errorf("Sample %d: expected X[0]=%v, got %v", i, expected[i], s.X[0])
		}
	}
}

func TestSamplesToTensors(t *testing.T) {
	samples := []Sample{
		{X: []float32{1, 2}, Y: []float32{0.5}},
		{X: []float32{3, 4}, Y: []float32{0.6}},
		{X: []float32{5, 6}, Y: []float32{0.7}},
	}

	x, y, err := samplesToTensors(samples)
	if err != nil {
		t.Fatalf("samplesToTensors error: %v", err)
	}

	if len(x.Shape) != 2 || x.Shape[0] != 3 || x.Shape[1] != 2 {
		t.Fatalf("Expected x shape [3, 2], got %v", x.Shape)
	}
	if len(y.Shape) != 2 || y.Shape[0] != 3 || y.Shape[1] != 1 {
		t.Fatalf("Expected y shape [3, 1], got %v", y.Shape)
	}

	// Verify data
	expectedX := []float32{1, 2, 3, 4, 5, 6}
	for i, v := range expectedX {
		if x.Data[i] != v {
			t.Errorf("x.Data[%d]: expected %v, got %v", i, v, x.Data[i])
		}
	}

	expectedY := []float32{0.5, 0.6, 0.7}
	for i, v := range expectedY {
		if y.Data[i] != v {
			t.Errorf("y.Data[%d]: expected %v, got %v", i, v, y.Data[i])
		}
	}
}

func TestRetrainSwapsWhenBetter(t *testing.T) {
	model := xorModel(t)
	holdout := xorHoldout()

	cfg := Config{
		InputDim:     2,
		Loss:         train.BCELoss(),
		Holdout:      holdout,
		BufferSize:   1024,
		RetrainEvery: 256,
		Epochs:       300,
		LearningRate: 0.9,
	}

	s, err := New(model, cfg)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	if s.Generation() != 1 {
		t.Fatalf("Expected initial generation 1, got %d", s.Generation())
	}

	// Get initial holdout loss
	initialLoss, err := s.holdoutLoss(s.current.Load().doc)
	if err != nil {
		t.Fatalf("Failed to get initial holdout loss: %v", err)
	}

	samples := xorSamples()
	s.retrain(samples)

	if s.Generation() != 2 {
		t.Fatalf("Expected generation 2 after retrain, got %d", s.Generation())
	}

	// Get new holdout loss
	newLoss, err := s.holdoutLoss(s.current.Load().doc)
	if err != nil {
		t.Fatalf("Failed to get new holdout loss: %v", err)
	}

	// New loss should be better (lower) than initial
	if newLoss >= initialLoss {
		t.Fatalf("Expected new loss %v < initial loss %v", newLoss, initialLoss)
	}
}

func TestGateRejectsWorseCandidate(t *testing.T) {
	model := xorModel(t)
	holdout := xorHoldout()
	samples := xorSamples()

	// Pre-train the model using train.Trainer until holdout loss < 0.2
	trainX, trainY, err := samplesToTensors(samples)
	if err != nil {
		t.Fatalf("Failed to convert samples to tensors: %v", err)
	}

	holdoutX, holdoutY, err := samplesToTensors(holdout)
	if err != nil {
		t.Fatalf("Failed to convert holdout to tensors: %v", err)
	}

	trainer := train.New(model, train.SGD(0.9), train.BCELoss())
	for epoch := 0; epoch < 100; epoch++ {
		_, err := trainer.Fit(trainX, trainY,
			train.Epochs(1),
			train.BatchSize(4),
			train.Shuffle(true),
			train.Seed(int64(epoch)),
		)
		if err != nil {
			t.Fatalf("Failed to fit model: %v", err)
		}

		metrics, err := trainer.Evaluate(holdoutX, holdoutY)
		if err != nil {
			t.Fatalf("Failed to evaluate model: %v", err)
		}

		if metrics.Loss < 0.2 {
			break
		}
	}

	// Now create the server with the pre-trained model
	cfg := Config{
		InputDim:     2,
		Loss:         train.BCELoss(),
		Holdout:      holdout,
		BufferSize:   1024,
		RetrainEvery: 256,
		Epochs:       300,
		LearningRate: 0.9,
	}

	s, err := New(model, cfg)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	if s.Generation() != 1 {
		t.Fatalf("Expected generation 1 after pre-training, got %d", s.Generation())
	}

	// Now create corrupted labels (all 0s, which is definitely wrong for XOR)
	badSamples := []Sample{
		{X: []float32{0, 0}, Y: []float32{0}},
		{X: []float32{0, 1}, Y: []float32{0}},
		{X: []float32{1, 0}, Y: []float32{0}},
		{X: []float32{1, 1}, Y: []float32{0}},
	}

	// Call retrain with bad labels
	s.retrain(badSamples)

	// Generation should still be 1 (swap rejected because bad labels hurt model)
	if s.Generation() != 1 {
		t.Fatalf("Expected generation 1 (swap rejected), got %d", s.Generation())
	}

	// swapRejectedTotal should be 1
	if s.metrics.swapRejectedTotal.Load() != 1 {
		t.Fatalf("Expected swapRejectedTotal 1, got %d", s.metrics.swapRejectedTotal.Load())
	}

	// Get predictions and verify they're unchanged
	pred1, gen1, err := s.Predict([]float32{0, 0})
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if gen1 != 1 {
		t.Fatalf("Expected gen 1, got %d", gen1)
	}

	// Verify they match what we got before
	pred2, gen2, err := s.Predict([]float32{0, 0})
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if gen2 != 1 {
		t.Fatalf("Expected gen 1, got %d", gen2)
	}

	if pred1[0] != pred2[0] {
		t.Fatalf("Predictions changed after rejected swap")
	}
}

func TestFeedbackEndpointAccepts(t *testing.T) {
	model := xorModel(t)
	holdout := xorHoldout()

	cfg := Config{
		InputDim: 2,
		Loss:     train.BCELoss(),
		Holdout:  holdout,
	}

	s, err := New(model, cfg)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	handler := s.Handler()

	body := map[string]interface{}{
		"x": []float32{0.5, 0.5},
		"y": []float32{0.5},
	}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/feedback", bytes.NewReader(bodyBytes))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusAccepted {
		t.Fatalf("Expected 202 Accepted, got %d", w.Code)
	}

	if s.metrics.feedbackTotal.Load() != 1 {
		t.Fatalf("Expected feedbackTotal 1, got %d", s.metrics.feedbackTotal.Load())
	}
}

func TestStartOnlineRequiresConfig(t *testing.T) {
	model := xorModel(t)

	// Test with missing Loss
	cfg := Config{
		InputDim: 2,
		Holdout:  xorHoldout(),
		Loss:     nil,
	}
	s, err := New(model, cfg)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err = s.StartOnline(ctx)
	if err == nil {
		t.Fatalf("Expected error when Loss is nil")
	}

	// Test with missing Holdout
	cfg2 := Config{
		InputDim: 2,
		Loss:     train.BCELoss(),
		Holdout:  nil,
	}
	s2, err := New(model, cfg2)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	err = s2.StartOnline(ctx)
	if err == nil {
		t.Fatalf("Expected error when Holdout is nil")
	}
}

func TestFeedbackDropsWhenChannelFull(t *testing.T) {
	model := xorModel(t)
	holdout := xorHoldout()

	cfg := Config{
		InputDim: 2,
		Loss:     train.BCELoss(),
		Holdout:  holdout,
	}

	s, err := New(model, cfg)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}

	// Replace feedback channel with small capacity (2) so we can fill it with 3 requests
	s.feedback = make(chan Sample, 2)

	handler := s.Handler()

	// Post 3 feedback requests; first 2 should succeed, 3rd should be dropped
	for i := 0; i < 3; i++ {
		body := map[string]interface{}{
			"x": []float32{0.5, 0.5},
			"y": []float32{0.5},
		}
		bodyBytes, _ := json.Marshal(body)

		req := httptest.NewRequest("POST", "/feedback", bytes.NewReader(bodyBytes))
		w := httptest.NewRecorder()

		handler.ServeHTTP(w, req)

		// All requests should return 202 (non-blocking send never blocks caller)
		if w.Code != http.StatusAccepted {
			t.Fatalf("Request %d: expected 202 Accepted, got %d", i+1, w.Code)
		}
	}

	// After 3 requests to a channel with capacity 2:
	// - feedbackTotal should be 2 (only successful sends increment it)
	// - feedbackDropped should be 1 (1 send went to default branch)
	if s.metrics.feedbackTotal.Load() != 2 {
		t.Fatalf("Expected feedbackTotal 2, got %d", s.metrics.feedbackTotal.Load())
	}

	if s.metrics.feedbackDropped.Load() != 1 {
		t.Fatalf("Expected feedbackDropped 1, got %d", s.metrics.feedbackDropped.Load())
	}
}
