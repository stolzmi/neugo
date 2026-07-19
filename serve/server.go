package serve

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sync"
	"sync/atomic"

	"neugo/nn"
	"neugo/train"
)

// Sample represents a training sample with input and target.
type Sample struct {
	X, Y []float32
}

// Config configures a Server.
type Config struct {
	InputDim           int          // required; length every input must have
	Loss               train.Loss   // required for online learning (Task 8); nil = serve-only
	Holdout            []Sample     // required for online learning; nil = serve-only
	BufferSize         int          // ring buffer capacity, default 1024
	RetrainEvery       int          // retrain after N feedback samples, default 256
	Epochs             int          // per retrain, default 5
	LearningRate       float32      // default 0.05
	MaxValLossIncrease float32      // gate slack, default 0 (candidate must be at least as good)
}

// modelVersion represents a generation of the model with a pool of clones.
type modelVersion struct {
	gen  uint64
	doc  []byte
	pool sync.Pool
}

// Server is a lock-free HTTP server for model inference.
// swapMu guards swapIn only — the predict path is lock-free and must NOT touch it.
type Server struct {
	current  *atomic.Pointer[modelVersion]
	cfg      Config
	metrics  *metrics
	swapMu   sync.Mutex // guards swapIn only; serializes model swaps to keep generations unique
	feedback chan Sample // buffered channel for online learning feedback
}

// New creates a new Server from a model and config.
func New(model *nn.SequentialModel, cfg Config) (*Server, error) {
	if cfg.InputDim <= 0 {
		return nil, fmt.Errorf("serve: InputDim must be positive")
	}

	// Apply defaults
	if cfg.BufferSize <= 0 {
		cfg.BufferSize = 1024
	}
	if cfg.RetrainEvery <= 0 {
		cfg.RetrainEvery = 256
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 5
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.05
	}
	// MaxValLossIncrease defaults to 0 (no slack)

	// Marshal the model
	doc, err := nn.Marshal(model)
	if err != nil {
		return nil, fmt.Errorf("serve: Marshal: %w", err)
	}

	// Create the initial model version
	ver := &modelVersion{
		gen: 1,
		doc: doc,
	}
	ver.pool.New = func() any {
		m, err := nn.Unmarshal(ver.doc)
		if err != nil {
			// This should never happen if the doc is valid (it was just marshaled)
			panic(fmt.Sprintf("serve: failed to unmarshal model: %v", err))
		}
		return m
	}

	ptr := &atomic.Pointer[modelVersion]{}
	ptr.Store(ver)

	return &Server{
		current:  ptr,
		cfg:      cfg,
		metrics:  &metrics{},
		feedback: make(chan Sample, 4096),
	}, nil
}

// Handler returns an http.Handler for the server routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", s.handlePredict)
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("POST /feedback", s.handleFeedback)
	return mux
}

// ListenAndServe starts the HTTP server on the given address.
func (s *Server) ListenAndServe(addr string) error {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	server := &http.Server{Handler: s.Handler()}
	return server.Serve(listener)
}

// Predict performs inference on the given input.
// Returns the output, model generation, and any error.
// The returned output slice is owned by the caller.
func (s *Server) Predict(x []float32) ([]float32, uint64, error) {
	if len(x) != s.cfg.InputDim {
		return nil, 0, fmt.Errorf("serve: input length %d does not match InputDim %d", len(x), s.cfg.InputDim)
	}

	// Load current model version
	ver := s.current.Load()

	// Get a clone from the pool
	modelAny := ver.pool.Get()
	model := modelAny.(*nn.SequentialModel)
	defer ver.pool.Put(model)

	// Create input tensor [1, InputDim]
	inputTensor, err := nn.NewTensorFromData(x, []int{1, s.cfg.InputDim})
	if err != nil {
		return nil, 0, fmt.Errorf("serve: NewTensorFromData: %w", err)
	}

	// Forward pass with Inference mode
	ctx := &nn.Context{Mode: nn.Inference}
	outputTensor, err := model.Forward(ctx, inputTensor)
	if err != nil {
		return nil, 0, fmt.Errorf("serve: Forward: %w", err)
	}

	// Copy output data into a fresh slice (caller owns it)
	output := make([]float32, len(outputTensor.Data))
	copy(output, outputTensor.Data)

	s.metrics.predictTotal.Add(1)

	return output, ver.gen, nil
}

// Generation returns the current model generation.
func (s *Server) Generation() uint64 {
	ver := s.current.Load()
	return ver.gen
}

// swapIn installs a new model generation from marshaled bytes.
// Lock serializes this to keep generations unique and monotonic.
func (s *Server) swapIn(doc []byte) {
	s.swapMu.Lock()
	defer s.swapMu.Unlock()

	oldVer := s.current.Load()
	newGen := oldVer.gen + 1

	// Create new model version
	newVer := &modelVersion{
		gen: newGen,
		doc: doc,
	}
	newVer.pool.New = func() any {
		m, err := nn.Unmarshal(newVer.doc)
		if err != nil {
			panic(fmt.Sprintf("serve: failed to unmarshal model: %v", err))
		}
		return m
	}

	// Atomically swap (old version stays valid for in-flight requests)
	s.current.Store(newVer)
	s.metrics.swapTotal.Add(1)
	s.metrics.modelGen.Store(newGen)
}

// handlePredict handles POST /predict requests.
func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Input []float32 `json:"input"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}

	output, gen, err := s.Predict(req.Input)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"output":    output,
		"model_gen": gen,
	})
}

// handleHealthz handles GET /healthz requests.
func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	gen := s.Generation()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"model_gen": gen,
	})
}

// handleFeedback handles POST /feedback requests.
// Validates X and Y lengths, then non-blocking send to the feedback channel.
func (s *Server) handleFeedback(w http.ResponseWriter, r *http.Request) {
	var req struct {
		X []float32 `json:"x"`
		Y []float32 `json:"y"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}

	// Validate X length
	if len(req.X) != s.cfg.InputDim {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("X length %d does not match InputDim %d", len(req.X), s.cfg.InputDim)})
		return
	}

	// Validate Y length (if holdout is set, use its Y dimension)
	if s.cfg.Holdout != nil && len(s.cfg.Holdout) > 0 {
		expectedYLen := len(s.cfg.Holdout[0].Y)
		if len(req.Y) != expectedYLen {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("Y length %d does not match holdout Y length %d", len(req.Y), expectedYLen)})
			return
		}
	}

	sample := Sample{X: req.X, Y: req.Y}

	// Non-blocking send: if channel is full, drop and increment feedbackDropped
	select {
	case s.feedback <- sample:
		// Successfully sent
		s.metrics.feedbackTotal.Add(1)
	default:
		// Channel full, drop the sample
		s.metrics.feedbackDropped.Add(1)
	}

	w.WriteHeader(http.StatusAccepted)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "accepted"})
}
