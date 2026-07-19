package tune

import (
	"context"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"
)

// Trial represents a single trial in the search.
type Trial struct {
	ID              int
	Params          Params
	Seed            int64
	asha            *asha // shared ASHA state; internal
	lastShouldPrune bool  // cached result of last Report decision; internal
}

// Report records an intermediate metric at a specific resource level (e.g., epoch count).
// If ASHA is configured, it evaluates whether the trial should be pruned based on
// how the reported value compares to other trials at the same resource level.
//
// Documented usage pattern:
//
//	for epoch := 1; epoch <= maxEpochs; epoch++ {
//		loss = trainOneEpoch()
//		tr.Report(epoch, loss)
//		if tr.ShouldPrune() {
//			return loss, nil // tuner marks the TrialResult Pruned
//		}
//	}
func (t *Trial) Report(resource int, value float64) {
	if t.asha == nil {
		t.lastShouldPrune = false
		return
	}
	t.lastShouldPrune = t.asha.report(resource, value)
}

// ShouldPrune returns true if the trial should be pruned based on the last Report decision.
// If ASHA is not configured, always returns false.
func (t *Trial) ShouldPrune() bool {
	return t.lastShouldPrune
}

// Objective is a function that evaluates a trial and returns a score.
// Lower scores are better by default; cfg.Maximize reverses this.
type Objective func(t *Trial) (float64, error)

// ASHAConfig is a placeholder for ASHA pruning (Task 12).
type ASHAConfig struct {
	MinResource     int
	MaxResource     int
	ReductionFactor int
}

// Config specifies the tuning configuration.
type Config struct {
	Trials   int
	Workers  int // <=0 → runtime.NumCPU()
	Seed     int64
	Maximize bool
	ASHA     *ASHAConfig // Task 12; nil = no pruning
}

// Run performs a random search across the space.
// Returns (results, nil) on success, (partialResults, ctx.Err()) on cancellation.
func Run(ctx context.Context, space *Space, obj Objective, cfg Config) (*Results, error) {
	if cfg.Workers <= 0 {
		cfg.Workers = runtime.NumCPU()
	}

	// Create shared ASHA instance if configured (Task 12).
	ashaInstance := newASHA(cfg.ASHA, cfg.Maximize)

	// Step 1: Sample all trial params upfront, sequentially, deterministically.
	r := rand.New(rand.NewSource(cfg.Seed))
	trials := make([]*Trial, cfg.Trials)
	for i := 0; i < cfg.Trials; i++ {
		trials[i] = &Trial{
			ID:     i,
			Params: space.Sample(r),
			Seed:   cfg.Seed + int64(i),
			asha:   ashaInstance,
		}
	}

	// Step 2: Create results array with disjoint indices (no mutex needed).
	// Pre-populate with ErrNotRun so cancelled (never-run) trials are distinguishable.
	results := make([]TrialResult, cfg.Trials)
	for i := 0; i < cfg.Trials; i++ {
		results[i].ID = i
		results[i].Err = ErrNotRun
	}

	// Step 3: Feed trials through a channel to workers.
	trialChan := make(chan *Trial, cfg.Workers)

	// Step 4: Start workers with WaitGroup.
	var wg sync.WaitGroup
	for w := 0; w < cfg.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case trial, ok := <-trialChan:
					if !ok {
						return
					}
					// Run the objective and record the result.
					start := time.Now()
					value, err := obj(trial)
					elapsed := time.Since(start)

					results[trial.ID] = TrialResult{
						ID:       trial.ID,
						Params:   trial.Params,
						Value:    value,
						Err:      err,
						Pruned:   trial.lastShouldPrune,
						Duration: elapsed,
					}
				}
			}
		}()
	}

	// Step 5: Feed trials into the channel.
	go func() {
		defer close(trialChan)
		for _, trial := range trials {
			select {
			case <-ctx.Done():
				return
			case trialChan <- trial:
			}
		}
	}()

	// Step 6: Wait for all workers to finish.
	wg.Wait()

	// Check if context was cancelled.
	if ctx.Err() != nil {
		sortResults(results, cfg.Maximize)
		return &Results{Trials: results}, ctx.Err()
	}

	// Step 7: Sort results (best-first per cfg.Maximize; errored and pruned trials after all scored ones).
	sortResults(results, cfg.Maximize)

	return &Results{Trials: results}, nil
}

// sortResults sorts the results array in-place.
// Best-first per Maximize; errored and pruned trials after all scored ones.
func sortResults(results []TrialResult, maximize bool) {
	// Partition: successful non-pruned trials first, then errored/pruned.
	good := 0
	for i := 0; i < len(results); i++ {
		if results[i].Err == nil && !results[i].Pruned {
			results[good], results[i] = results[i], results[good]
			good++
		}
	}

	// Sort the good trials using sort.Slice.
	if maximize {
		// Sort descending by value.
		sort.Slice(results[:good], func(i, j int) bool {
			return results[i].Value > results[j].Value
		})
	} else {
		// Sort ascending by value.
		sort.Slice(results[:good], func(i, j int) bool {
			return results[i].Value < results[j].Value
		})
	}
}
