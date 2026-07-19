package tune

import (
	"math"
	"sync"
)

// asha represents the shared ASHA successive-halving pruning state across all trials.
// It manages rung decisions for early stopping via successive halving.
type asha struct {
	cfg      *ASHAConfig
	maximize bool
	rungs    map[int][]float64 // resource -> slice of recorded values
	mu       sync.Mutex
}

// newASHA creates a new ASHA instance from a config, or nil if cfg is nil.
func newASHA(cfg *ASHAConfig, maximize bool) *asha {
	if cfg == nil {
		return nil
	}

	return &asha{
		cfg:      cfg,
		maximize: maximize,
		rungs:    make(map[int][]float64),
	}
}

// eta returns the reduction factor, defaulting to 3 if <= 1.
func (a *asha) eta() int {
	if a == nil {
		return 3
	}
	eta := a.cfg.ReductionFactor
	if eta <= 1 {
		eta = 3
	}
	return eta
}

// isRung checks if a resource equals a rung exactly.
// Rungs are at MinResource * η^k for k = 0, 1, 2, ... up to MaxResource.
func (a *asha) isRung(resource int) bool {
	if a == nil || resource < a.cfg.MinResource || resource > a.cfg.MaxResource {
		return false
	}

	eta := a.eta()

	// Check if resource = MinResource * η^k for some k >= 0
	current := a.cfg.MinResource
	for current <= a.cfg.MaxResource {
		if current == resource {
			return true
		}
		current *= eta
	}
	return false
}

// decide makes a promotion decision for the last value in the values slice.
// Returns true if the trial should be pruned, false if promoted.
// Decision: promoted iff the last value is within the best ceil(n/η) of all values,
// where direction is determined by the maximize flag.
// With n < η observations, promotes by default (async ASHA).
func (a *asha) decide(maximize bool, values []float64) bool {
	n := len(values)
	if n == 0 {
		return false
	}

	eta := a.eta()

	// If fewer than η observations, promote by default (async ASHA)
	if n < eta {
		return false
	}

	// Find the threshold: the ceil(n/η)-th best value (1-indexed becomes 0-indexed)
	keepCount := int(math.Ceil(float64(n) / float64(eta)))
	if keepCount > n {
		keepCount = n
	}

	newValue := values[n-1]

	// Count how many values are at least as good as newValue (including ties)
	betterOrEqual := 0
	if maximize {
		for _, v := range values {
			if v >= newValue {
				betterOrEqual++
			}
		}
	} else {
		for _, v := range values {
			if v <= newValue {
				betterOrEqual++
			}
		}
	}

	// Prune if the value is NOT in the top keepCount
	// i.e., if more than keepCount values are at least as good
	return betterOrEqual > keepCount
}

// report records a value at a resource and decides whether to prune.
// If resource is not a rung, returns false (never prunes).
// Returns true if the trial should be pruned, false if promoted.
func (a *asha) report(resource int, value float64) bool {
	if a == nil || !a.isRung(resource) {
		// Non-rung resources never prune, nil config never prunes
		return false
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Record the value at this rung
	a.rungs[resource] = append(a.rungs[resource], value)
	values := a.rungs[resource]

	// Decide promotion using stored maximize flag
	return a.decide(a.maximize, values)
}
