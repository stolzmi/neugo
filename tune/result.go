package tune

import (
	"bytes"
	"fmt"
	"text/tabwriter"
	"time"
)

// TrialResult represents the outcome of a single trial.
type TrialResult struct {
	ID       int
	Params   Params
	Value    float64
	Err      error
	Pruned   bool
	Duration time.Duration
}

// Results holds all trial results, sorted best-first.
type Results struct {
	Trials []TrialResult
}

// Best returns the best trial result.
// Assumes Trials are already sorted (best-first).
func (r *Results) Best() TrialResult {
	if len(r.Trials) == 0 {
		return TrialResult{}
	}
	return r.Trials[0]
}

// Top returns the top k trial results.
// Assumes Trials are already sorted (best-first).
func (r *Results) Top(k int) []TrialResult {
	if k < 0 {
		k = 0
	}
	if k > len(r.Trials) {
		k = len(r.Trials)
	}
	return r.Trials[:k]
}

// String returns a text table representation of the results.
func (r *Results) String() string {
	var buf bytes.Buffer
	w := tabwriter.NewWriter(&buf, 0, 0, 2, ' ', 0)

	// Header
	fmt.Fprintf(w, "Rank\tID\tValue\tParams\n")

	// Rows
	for rank, tr := range r.Trials {
		paramsStr := paramsToString(tr.Params)
		fmt.Fprintf(w, "%d\t%d\t%v\t%s\n", rank+1, tr.ID, tr.Value, paramsStr)
	}

	w.Flush()
	return buf.String()
}

// paramsToString converts Params to a string representation.
func paramsToString(p Params) string {
	if len(p) == 0 {
		return ""
	}
	var buf bytes.Buffer
	first := true
	for k, v := range p {
		if !first {
			buf.WriteString(", ")
		}
		fmt.Fprintf(&buf, "%s=", k)
		switch val := v.(type) {
		case float64:
			fmt.Fprintf(&buf, "%.4g", val)
		case int:
			fmt.Fprintf(&buf, "%d", val)
		case string:
			fmt.Fprintf(&buf, "%s", val)
		default:
			fmt.Fprintf(&buf, "%v", val)
		}
		first = false
	}
	return buf.String()
}
