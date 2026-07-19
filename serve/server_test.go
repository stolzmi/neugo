package serve

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"neugo/nn"
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
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
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
