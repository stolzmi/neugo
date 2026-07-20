package serve

import (
	"context"
	"fmt"

	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

// ringBuffer is a fixed-capacity circular buffer for samples.
// Push overwrites the oldest sample when full.
type ringBuffer struct {
	samples []Sample
	cap     int
	index   int // next write position
	count   int // number of samples (min(cap, index))
}

// newRingBuffer creates a fixed-capacity ring buffer.
func newRingBuffer(cap int) *ringBuffer {
	return &ringBuffer{
		samples: make([]Sample, cap),
		cap:     cap,
		index:   0,
		count:   0,
	}
}

// Push adds a sample, overwriting the oldest if buffer is full.
func (rb *ringBuffer) Push(s Sample) {
	rb.samples[rb.index] = s
	rb.index = (rb.index + 1) % rb.cap
	if rb.count < rb.cap {
		rb.count++
	}
}

// Snapshot returns a copy of all samples currently in the buffer in order.
func (rb *ringBuffer) Snapshot() []Sample {
	if rb.count < rb.cap {
		// Not wrapped yet; samples are at 0..count-1
		result := make([]Sample, rb.count)
		copy(result, rb.samples[:rb.count])
		return result
	}
	// Wrapped; oldest is at rb.index, going forward
	result := make([]Sample, rb.cap)
	for i := 0; i < rb.cap; i++ {
		result[i] = rb.samples[(rb.index+i)%rb.cap]
	}
	return result
}

// samplesToTensors converts a slice of samples into x and y tensors.
// x has shape [n, dimX], y has shape [n, dimY]
func samplesToTensors(samples []Sample) (*nn.Tensor, *nn.Tensor, error) {
	if len(samples) == 0 {
		return nil, nil, fmt.Errorf("serve: cannot convert empty samples to tensors")
	}

	dimX := len(samples[0].X)
	dimY := len(samples[0].Y)

	for i, s := range samples {
		if len(s.X) != dimX {
			return nil, nil, fmt.Errorf("serve: sample %d X length %d != %d", i, len(s.X), dimX)
		}
		if len(s.Y) != dimY {
			return nil, nil, fmt.Errorf("serve: sample %d Y length %d != %d", i, len(s.Y), dimY)
		}
	}

	x := nn.NewTensor([]int{len(samples), dimX})
	y := nn.NewTensor([]int{len(samples), dimY})

	for i, s := range samples {
		copy(x.Data[i*dimX:(i+1)*dimX], s.X)
		copy(y.Data[i*dimY:(i+1)*dimY], s.Y)
	}

	return x, y, nil
}

// holdoutLoss evaluates a marshaled model on the holdout set and returns the loss.
func (s *Server) holdoutLoss(doc []byte) (float32, error) {
	if s.cfg.Holdout == nil {
		return 0, fmt.Errorf("serve: holdout set not configured")
	}
	if s.cfg.Loss == nil {
		return 0, fmt.Errorf("serve: loss not configured")
	}

	model, err := nn.Unmarshal(doc)
	if err != nil {
		return 0, fmt.Errorf("serve: unmarshal: %w", err)
	}

	x, y, err := samplesToTensors(s.cfg.Holdout)
	if err != nil {
		return 0, fmt.Errorf("serve: samplesToTensors: %w", err)
	}

	trainer := train.New(model, train.SGD(s.cfg.LearningRate), s.cfg.Loss)
	metrics, err := trainer.Evaluate(x, y)
	if err != nil {
		return 0, fmt.Errorf("serve: Evaluate: %w", err)
	}

	return metrics.Loss, nil
}

// retrain trains a new model on the given samples and swaps it in if it passes the gate.
func (s *Server) retrain(samples []Sample) {
	// Read current model version once at entry
	curVer := s.current.Load()
	curLoss, err := s.holdoutLoss(curVer.doc)
	if err != nil {
		// If we can't evaluate the current model, abort
		return
	}

	// Unmarshal candidate model
	candidate, err := nn.Unmarshal(curVer.doc)
	if err != nil {
		return
	}

	// Convert samples to tensors
	x, y, err := samplesToTensors(samples)
	if err != nil {
		return
	}

	// Train the candidate
	trainer := train.New(candidate, train.SGD(s.cfg.LearningRate), s.cfg.Loss)
	batchSize := 32
	if len(samples) < batchSize {
		batchSize = len(samples)
	}

	_, err = trainer.Fit(x, y,
		train.Epochs(s.cfg.Epochs),
		train.BatchSize(batchSize),
		train.Shuffle(true),
		train.Seed(int64(curVer.gen)),
	)
	if err != nil {
		return
	}

	// Evaluate on holdout set
	candDoc, err := nn.Marshal(candidate)
	if err != nil {
		return
	}

	candLoss, err := s.holdoutLoss(candDoc)
	if err != nil {
		return
	}

	// Apply gate: candidate loss <= current loss + MaxValLossIncrease
	if candLoss <= curLoss+s.cfg.MaxValLossIncrease {
		s.swapIn(candDoc)
	} else {
		s.metrics.swapRejectedTotal.Add(1)
	}
}

// StartOnline starts the online learning goroutine.
// It receives feedback samples and periodically retrains the model.
// Returns error if cfg.Loss or cfg.Holdout is nil.
func (s *Server) StartOnline(ctx context.Context) error {
	if s.cfg.Loss == nil {
		return fmt.Errorf("serve: Loss not configured; cannot start online learning")
	}
	if s.cfg.Holdout == nil {
		return fmt.Errorf("serve: Holdout not configured; cannot start online learning")
	}

	go s.onlineTrainer(ctx)
	return nil
}

// onlineTrainer is the main online learning loop.
func (s *Server) onlineTrainer(ctx context.Context) {
	rb := newRingBuffer(s.cfg.BufferSize)
	count := 0

	for {
		select {
		case <-ctx.Done():
			return
		case sample := <-s.feedback:
			rb.Push(sample)
			count++

			// Check if we should retrain
			if count >= s.cfg.RetrainEvery {
				snapshot := rb.Snapshot()
				if len(snapshot) > 0 {
					s.retrain(snapshot)
				}
				count = 0
			}
		}
	}
}
