package tune

import (
	"context"
	"sync/atomic"
	"testing"
)

func TestRungPromotionMath(t *testing.T) {
	// Unit test on the asha struct: rung already holds {1..9}, η=3:
	// decide(1.5) → promote (top third); decide(8.0) → prune. Minimize direction.
	a := &asha{
		cfg: &ASHAConfig{
			MinResource:     1,
			MaxResource:     81,
			ReductionFactor: 3,
		},
		maximize: false, // minimize direction
		rungs:    make(map[int][]float64),
	}

	// Pre-populate a rung with {1..9}
	resource := 9
	a.rungs[resource] = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}

	// decide(1.5) with {1..9} and η=3:
	// n=9, ceil(9/3)=3 best values = {1,2,3}
	// 1.5 should be promoted (within top 3)
	shouldPrune := a.decide(false, append(a.rungs[resource], 1.5))
	if shouldPrune {
		t.Errorf("decide(1.5) with {1..9} should promote, got prune")
	}

	// decide(8.0) with {1..9,1.5} and η=3:
	// n=10, ceil(10/3)=4 best values = {1,1.5,2,3}
	// 8.0 is not in top 4, so should prune
	shouldPrune = a.decide(false, append(a.rungs[resource], 1.5, 8.0))
	if !shouldPrune {
		t.Errorf("decide(8.0) with {1..9,1.5} should prune, got promote")
	}
}

func TestNonRungResourceNeverPrunes(t *testing.T) {
	a := &asha{
		cfg: &ASHAConfig{
			MinResource:     1,
			MaxResource:     81,
			ReductionFactor: 3,
		},
		maximize: false,
		rungs:    make(map[int][]float64),
	}

	// Report at a non-rung resource (e.g., 2, which is between 1 and 3)
	shouldPrune := a.report(2, 10.0)
	if shouldPrune {
		t.Errorf("Non-rung resource should never prune, got prune")
	}

	// Report at another non-rung resource
	shouldPrune = a.report(5, 5.0)
	if shouldPrune {
		t.Errorf("Non-rung resource should never prune, got prune")
	}
}

func TestNoASHAConfigNeverPrunes(t *testing.T) {
	// With cfg.ASHA == nil, Report is a no-op and ShouldPrune always returns false.
	space := NewSpace().Float("x", -1, 1)
	obj := func(trial *Trial) (float64, error) {
		x := trial.Params.Float("x")
		for epoch := 1; epoch <= 10; epoch++ {
			loss := float64(epoch) * x // arbitrary loss
			trial.Report(epoch, loss)
			if trial.ShouldPrune() {
				t.Errorf("ShouldPrune should always be false with no ASHA config")
			}
		}
		return float64(10) * x, nil
	}

	cfg := Config{
		Trials:  10,
		Workers: 2,
		Seed:    42,
		ASHA:    nil, // no ASHA config
	}

	results, err := Run(context.Background(), space, obj, cfg)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if results == nil || len(results.Trials) == 0 {
		t.Fatal("No results returned")
	}

	// No trials should be marked as pruned
	for _, tr := range results.Trials {
		if tr.Pruned {
			t.Errorf("Trial %d marked as pruned, but ASHA is disabled", tr.ID)
		}
	}
}

func TestAshaPrunesBadTrialsAndSavesWork(t *testing.T) {
	// Space: Float("x", -1, 1). Objective simulates 81 epochs:
	//   good (x>0): loss = 1/float64(epoch); bad (x<=0): loss = 10.
	//   Reports every epoch, honors ShouldPrune (returns early).
	// ASHA{1, 81, 3}, 100 trials, seed 7. Assert:
	//   Best().Value < 0.1; ≥30% of bad-x trials Pruned;
	//   total epochs executed (count via atomic in objective) < 100*81/2.

	var epochsExecuted int64
	space := NewSpace().Float("x", -1, 1)
	obj := func(trial *Trial) (float64, error) {
		x := trial.Params.Float("x")
		maxEpochs := 81
		loss := 0.0

		for epoch := 1; epoch <= maxEpochs; epoch++ {
			atomic.AddInt64(&epochsExecuted, 1)

			if x > 0 {
				loss = 1.0 / float64(epoch)
			} else {
				loss = 10.0
			}

			trial.Report(epoch, loss)
			if trial.ShouldPrune() {
				return loss, nil
			}
		}
		return loss, nil
	}

	cfg := Config{
		Trials:  100,
		Workers: 4,
		Seed:    7,
		ASHA: &ASHAConfig{
			MinResource:     1,
			MaxResource:     81,
			ReductionFactor: 3,
		},
	}

	results, err := Run(context.Background(), space, obj, cfg)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	best := results.Best()
	if best.Value >= 0.1 {
		t.Errorf("Best value %.6f >= 0.1", best.Value)
	}

	// Count pruned trials where x <= 0
	badTrialsPruned := 0
	badTrialsTotal := 0
	for _, tr := range results.Trials {
		if tr.Err != nil {
			continue
		}
		x := tr.Params.Float("x")
		if x <= 0 {
			badTrialsTotal++
			if tr.Pruned {
				badTrialsPruned++
			}
		}
	}

	if badTrialsTotal == 0 {
		t.Errorf("No bad trials found (x <= 0)")
	}

	pruneRate := float64(badTrialsPruned) / float64(badTrialsTotal)
	if pruneRate < 0.3 {
		t.Errorf("Bad trial prune rate %.2f < 0.3 (need >=30%%)", pruneRate)
	}

	// Check total epochs
	totalEpochs := atomic.LoadInt64(&epochsExecuted)
	maxEpochs := 100 * 81 / 2
	if totalEpochs >= int64(maxEpochs) {
		t.Errorf("Total epochs %d >= %d (should save >50%% via pruning)", totalEpochs, maxEpochs)
	}
}
