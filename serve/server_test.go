package serve

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stolzmi/neugo/nn"
)

func tinyModel(t *testing.T) *nn.SequentialModel {
	rng := rand.New(rand.NewSource(42))
	model, err := nn.Sequential([]int{1, 2},
		nn.Linear(rng, 2, 4, nil),
		nn.ReLU(),
		nn.Linear(rng, 4, 1, nil),
		nn.Sigmoid(),
	)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	return model
}

func newTestServer(t *testing.T) *Server {
	model := tinyModel(t)
	cfg := Config{InputDim: 2}
	s, err := New(model, cfg)
	if err != nil {
		t.Fatalf("Failed to create server: %v", err)
	}
	return s
}

func TestPredictHTTP(t *testing.T) {
	s := newTestServer(t)
	handler := s.Handler()

	body := map[string][]float32{"input": {1, 0}}
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/predict", bytes.NewReader(bodyBytes))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	output, ok := resp["output"].([]interface{})
	if !ok || len(output) != 1 {
		t.Fatalf("Expected 1 output, got %v", output)
	}

	gen, ok := resp["model_gen"].(float64)
	if !ok || gen != 1 {
		t.Fatalf("Expected model_gen 1, got %v", gen)
	}
}

func TestPredictWrongLength(t *testing.T) {
	s := newTestServer(t)
	handler := s.Handler()

	body := map[string][]float32{"input": {1}} // Wrong length, should be 2
	bodyBytes, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/predict", bytes.NewReader(bodyBytes))
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("Expected 400, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to unmarshal error response: %v", err)
	}

	if _, ok := resp["error"]; !ok {
		t.Fatalf("Expected error field in response")
	}
}

func TestHealthz(t *testing.T) {
	s := newTestServer(t)
	handler := s.Handler()

	req := httptest.NewRequest("GET", "/healthz", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	gen, ok := resp["model_gen"].(float64)
	if !ok || gen != 1 {
		t.Fatalf("Expected model_gen 1, got %v", gen)
	}
}

func TestSwapBumpsGeneration(t *testing.T) {
	s := newTestServer(t)

	// Get initial prediction
	want, gen1, err := s.Predict([]float32{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if gen1 != 1 {
		t.Fatalf("Expected initial generation 1, got %d", gen1)
	}

	// Swap in a different model
	model2 := tinyModel(t)
	doc, err := nn.Marshal(model2)
	if err != nil {
		t.Fatalf("Failed to marshal model: %v", err)
	}
	s.swapIn(doc)

	// Get new prediction
	got, gen2, err := s.Predict([]float32{1, 0})
	if err != nil {
		t.Fatal(err)
	}

	if gen2 != 2 {
		t.Fatalf("Expected generation 2 after swap, got %d", gen2)
	}

	// Outputs should be different (very unlikely to be the same)
	if len(got) != len(want) || len(got) != 1 {
		t.Fatalf("Unexpected output shape")
	}
}

func TestPredictConcurrent(t *testing.T) {
	s := newTestServer(t)
	want, _, err := s.Predict([]float32{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				got, _, err := s.Predict([]float32{1, 0})
				if err != nil || got[0] != want[0] {
					t.Errorf("racy or failed prediction: %v %v", got, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestSwapConcurrentGenerationsUnique(t *testing.T) {
	s := newTestServer(t)

	// Capture the initial doc from the server's current model
	initialVer := s.current.Load()
	doc := initialVer.doc

	// Verify we start at generation 1
	if s.Generation() != 1 {
		t.Fatalf("Expected initial generation 1, got %d", s.Generation())
	}

	// Run 8 goroutines, each calling swapIn 25 times (200 total)
	const numGoroutines = 8
	const swapsPerGoroutine = 25
	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < swapsPerGoroutine; j++ {
				s.swapIn(doc)
			}
		}()
	}

	wg.Wait()

	// After all swaps, generation should be 1 + 200 = 201
	expectedGen := uint64(1 + numGoroutines*swapsPerGoroutine)
	finalGen := s.Generation()
	if finalGen != expectedGen {
		t.Fatalf("Expected generation %d, got %d", expectedGen, finalGen)
	}

	// Verify metrics.modelGen matches
	metricsGen := s.metrics.modelGen.Load()
	if metricsGen != expectedGen {
		t.Fatalf("Expected metrics.modelGen %d, got %d", expectedGen, metricsGen)
	}
}

func TestRollbackRestoresPreviousGeneration(t *testing.T) {
	s := newTestServer(t)

	// Get initial prediction from gen 1 model
	input := []float32{1, 0}
	original, gen1, err := s.Predict(input)
	if err != nil {
		t.Fatal(err)
	}
	if gen1 != 1 {
		t.Fatalf("Expected generation 1, got %d", gen1)
	}

	// Swap in a different model (gen 2)
	model2 := tinyModel(t)
	doc2, err := nn.Marshal(model2)
	if err != nil {
		t.Fatalf("Failed to marshal model: %v", err)
	}
	s.swapIn(doc2)

	// Verify we're on gen 2
	swapped, gen2, err := s.Predict(input)
	if err != nil {
		t.Fatal(err)
	}
	if gen2 != 2 {
		t.Fatalf("Expected generation 2 after swap, got %d", gen2)
	}

	// Rollback to gen 1
	gen3, err := s.Rollback()
	if err != nil {
		t.Fatalf("Rollback failed: %v", err)
	}
	if gen3 != 3 {
		t.Fatalf("Expected generation 3 after rollback, got %d", gen3)
	}

	// Verify prediction returns to original values (gen 3 model is same as gen 1)
	restored, gen3Check, err := s.Predict(input)
	if err != nil {
		t.Fatal(err)
	}
	if gen3Check != 3 {
		t.Fatalf("Expected generation 3, got %d", gen3Check)
	}
	if restored[0] != original[0] {
		t.Fatalf("Expected restored output %v, got %v", original, restored)
	}

	// Verify current and previous are exchanged:
	// current should be gen3 (the restored model), previous should be gen2 (the swapped model)
	currentVer := s.current.Load()
	if currentVer.gen != 3 {
		t.Fatalf("Expected current gen 3, got %d", currentVer.gen)
	}

	// Rollback again - should go back to gen 2
	gen4, err := s.Rollback()
	if err != nil {
		t.Fatalf("Second rollback failed: %v", err)
	}
	if gen4 != 4 {
		t.Fatalf("Expected generation 4 after second rollback, got %d", gen4)
	}

	rerollbacked, gen4Check, err := s.Predict(input)
	if err != nil {
		t.Fatal(err)
	}
	if gen4Check != 4 {
		t.Fatalf("Expected generation 4, got %d", gen4Check)
	}
	if rerollbacked[0] != swapped[0] {
		t.Fatalf("Expected rerollbacked output %v, got %v", swapped, rerollbacked)
	}
}

func TestRollbackWithoutPrevious(t *testing.T) {
	s := newTestServer(t)
	handler := s.Handler()

	// Fresh server has no previous version, should return 409
	req := httptest.NewRequest("POST", "/admin/rollback", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusConflict {
		t.Fatalf("Expected 409 Conflict, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if _, ok := resp["error"]; !ok {
		t.Fatalf("Expected error field in response")
	}
}

func TestMetricsEndpoint(t *testing.T) {
	s := newTestServer(t)
	handler := s.Handler()

	// Make a prediction to increment predictTotal
	body := map[string][]float32{"input": {1, 0}}
	bodyBytes, _ := json.Marshal(body)
	req := httptest.NewRequest("POST", "/predict", bytes.NewReader(bodyBytes))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Now request metrics
	req = httptest.NewRequest("GET", "/metrics", nil)
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d", w.Code)
	}

	// Check Content-Type
	contentType := w.Header().Get("Content-Type")
	if contentType != "text/plain; version=0.0.4" {
		t.Fatalf("Expected 'text/plain; version=0.0.4', got '%s'", contentType)
	}

	// Check body contains required metrics
	body_str := w.Body.String()
	if !bytes.Contains(w.Body.Bytes(), []byte("neugo_predict_total")) {
		t.Fatalf("Expected 'neugo_predict_total' in metrics, got:\n%s", body_str)
	}
	if !bytes.Contains(w.Body.Bytes(), []byte("neugo_model_generation")) {
		t.Fatalf("Expected 'neugo_model_generation' in metrics, got:\n%s", body_str)
	}
}
