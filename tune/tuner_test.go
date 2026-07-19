package tune

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"
)

func TestRunFindsMinimum(t *testing.T) {
	space := NewSpace().Float("x", -10, 10)
	obj := func(trial *Trial) (float64, error) {
		x := trial.Params.Float("x")
		return (x - 3) * (x - 3), nil
	}
	cfg := Config{
		Trials:   500,
		Workers:  8,
		Seed:     42,
		Maximize: false,
	}
	results, err := Run(context.Background(), space, obj, cfg)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	best := results.Best()
	if best.Value >= 0.05 {
		t.Errorf("Best value %.6f >= 0.05", best.Value)
	}
	x := best.Params.Float("x")
	if math.Abs(x-3) >= 0.25 {
		t.Errorf("Best x %.6f not close to 3 (diff %.6f >= 0.25)", x, math.Abs(x-3))
	}
}

func TestRunIsParallel(t *testing.T) {
	space := NewSpace().Float("x", 0, 1)
	obj := func(trial *Trial) (float64, error) {
		time.Sleep(20 * time.Millisecond)
		x := trial.Params.Float("x")
		return x, nil
	}
	cfg := Config{
		Trials:  32,
		Workers: 8,
		Seed:    42,
	}
	start := time.Now()
	results, err := Run(context.Background(), space, obj, cfg)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if results == nil || len(results.Trials) == 0 {
		t.Fatal("No results returned")
	}
	// Serial time would be 32 * 20ms = 640ms
	// Parallel with 8 workers should be much faster; assert < 320ms (half of serial)
	if elapsed >= 320*time.Millisecond {
		t.Errorf("Run took %v, expected < 320ms (looks serial)", elapsed)
	}
}

func TestRunDeterministicParams(t *testing.T) {
	space := NewSpace().Float("x", -10, 10)
	obj := func(trial *Trial) (float64, error) {
		x := trial.Params.Float("x")
		return x * x, nil
	}
	cfg := Config{
		Trials:  50,
		Workers: 4,
		Seed:    123,
	}
	results1, err1 := Run(context.Background(), space, obj, cfg)
	if err1 != nil {
		t.Fatalf("First Run failed: %v", err1)
	}
	results2, err2 := Run(context.Background(), space, obj, cfg)
	if err2 != nil {
		t.Fatalf("Second Run failed: %v", err2)
	}

	// Check Best().Params are identical
	best1 := results1.Best()
	best2 := results2.Best()
	if best1.ID != best2.ID {
		t.Errorf("Best trial ID mismatch: %d vs %d", best1.ID, best2.ID)
	}
	if best1.Params.Float("x") != best2.Params.Float("x") {
		t.Errorf("Best Params.Float(x) mismatch: %v vs %v", best1.Params.Float("x"), best2.Params.Float("x"))
	}

	// Check that all trial params are the same (DeepEqual for Params)
	if len(results1.Trials) != len(results2.Trials) {
		t.Fatalf("Trial count mismatch: %d vs %d", len(results1.Trials), len(results2.Trials))
	}
	for i, tr1 := range results1.Trials {
		tr2 := results2.Trials[i]
		if tr1.ID != tr2.ID {
			t.Errorf("Trial %d ID mismatch: %d vs %d", i, tr1.ID, tr2.ID)
		}
		if tr1.Params.Float("x") != tr2.Params.Float("x") {
			t.Errorf("Trial %d Params mismatch: %v vs %v", i, tr1.Params, tr2.Params)
		}
	}
}

func TestObjectiveErrorRecorded(t *testing.T) {
	space := NewSpace().Float("x", -5, 5)
	obj := func(trial *Trial) (float64, error) {
		x := trial.Params.Float("x")
		if x < 0 {
			return 0, errors.New("x must be non-negative")
		}
		return x, nil
	}
	cfg := Config{
		Trials:  50,
		Workers: 4,
		Seed:    42,
	}
	results, err := Run(context.Background(), space, obj, cfg)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Check that Best() has no error (should be among successful trials)
	best := results.Best()
	if best.Err != nil {
		t.Errorf("Best trial has error: %v", best.Err)
	}

	// Check that some trials recorded errors
	hasError := false
	for _, tr := range results.Trials {
		if tr.Err != nil {
			hasError = true
			break
		}
	}
	if !hasError {
		t.Errorf("Expected some trials to have errors")
	}
}

func TestRunHonorsContext(t *testing.T) {
	space := NewSpace().Float("x", 0, 1)
	obj := func(trial *Trial) (float64, error) {
		time.Sleep(100 * time.Millisecond)
		return trial.Params.Float("x"), nil
	}
	cfg := Config{
		Trials:  100,
		Workers: 4,
		Seed:    42,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	results, err := Run(ctx, space, obj, cfg)
	if err == nil {
		t.Errorf("Run should have returned context error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Expected context.DeadlineExceeded, got %v", err)
	}
	if results == nil {
		t.Errorf("Results should be non-nil on cancellation")
	}

	// Key assertion: results must be sorted and never-run trials must be distinguishable.
	// If any trial completed successfully (Err == nil && !Pruned), Best() must return one of those.
	// Otherwise, all trials must have ErrNotRun.
	completedCount := 0
	for _, tr := range results.Trials {
		if tr.Err == nil && !tr.Pruned {
			completedCount++
		}
	}

	if completedCount > 0 {
		// At least one trial completed; Best() must return a completed trial.
		best := results.Best()
		if best.Err != nil {
			t.Errorf("Best() returned a trial with Err=%v, but completed trials exist", best.Err)
		}
		if best.Pruned {
			t.Errorf("Best() returned a pruned trial, but non-pruned completed trials exist")
		}
	} else {
		// No trials completed; all must have ErrNotRun.
		for _, tr := range results.Trials {
			if !errors.Is(tr.Err, ErrNotRun) {
				t.Errorf("Trial %d has Err=%v, expected ErrNotRun (or nil only if completed)", tr.ID, tr.Err)
			}
		}
	}
}

func TestMaximize(t *testing.T) {
	space := NewSpace().Float("x", -10, 10)
	obj := func(trial *Trial) (float64, error) {
		x := trial.Params.Float("x")
		return -((x - 3) * (x - 3)), nil // negative of (x-3)^2
	}
	cfg := Config{
		Trials:   500,
		Workers:  8,
		Seed:     42,
		Maximize: true,
	}
	results, err := Run(context.Background(), space, obj, cfg)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	best := results.Best()
	if best.Value <= -0.05 { // maximizing negative means Best.Value should be close to 0 (the max)
		t.Errorf("Best value %.6f (should be close to 0 for maximize)", best.Value)
	}
	x := best.Params.Float("x")
	if math.Abs(x-3) >= 0.25 {
		t.Errorf("Best x %.6f not close to 3 (diff %.6f >= 0.25)", x, math.Abs(x-3))
	}
}

func TestRunRejectsInvalidASHAConfig(t *testing.T) {
	space := NewSpace().Float("x", 0, 1)
	obj := func(trial *Trial) (float64, error) {
		return trial.Params.Float("x"), nil
	}

	// Test 1: MinResource <= 0 should be rejected
	cfg := Config{
		Trials:  10,
		Workers: 2,
		Seed:    42,
		ASHA: &ASHAConfig{
			MinResource:     0, // invalid: <= 0
			MaxResource:     81,
			ReductionFactor: 3,
		},
	}
	results, err := Run(context.Background(), space, obj, cfg)
	if err == nil {
		t.Errorf("Run should reject MinResource <= 0, got nil error")
	}
	if results != nil {
		t.Errorf("Run should return nil results on invalid config, got %v", results)
	}

	// Test 2: MaxResource < MinResource should be rejected
	cfg = Config{
		Trials:  10,
		Workers: 2,
		Seed:    42,
		ASHA: &ASHAConfig{
			MinResource:     81, // MinResource > MaxResource
			MaxResource:     27, // invalid
			ReductionFactor: 3,
		},
	}
	results, err = Run(context.Background(), space, obj, cfg)
	if err == nil {
		t.Errorf("Run should reject MaxResource < MinResource, got nil error")
	}
	if results != nil {
		t.Errorf("Run should return nil results on invalid config, got %v", results)
	}

	// Test 3: Negative MinResource should be rejected
	cfg = Config{
		Trials:  10,
		Workers: 2,
		Seed:    42,
		ASHA: &ASHAConfig{
			MinResource:     -1, // invalid: negative
			MaxResource:     81,
			ReductionFactor: 3,
		},
	}
	results, err = Run(context.Background(), space, obj, cfg)
	if err == nil {
		t.Errorf("Run should reject negative MinResource, got nil error")
	}
	if results != nil {
		t.Errorf("Run should return nil results on invalid config, got %v", results)
	}
}
