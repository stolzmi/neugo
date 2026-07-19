package serve

import (
	"fmt"
	"io"
	"sync/atomic"
)

type metrics struct {
	predictTotal      atomic.Int64
	feedbackTotal     atomic.Int64
	feedbackDropped   atomic.Int64
	swapTotal         atomic.Int64
	swapRejectedTotal atomic.Int64
	modelGen          atomic.Uint64
}

func (m *metrics) writePrometheus(w io.Writer) {
	// Write all metrics in order with TYPE declarations
	fmt.Fprintf(w, "# TYPE neugo_predict_total counter\n")
	fmt.Fprintf(w, "neugo_predict_total %d\n", m.predictTotal.Load())

	fmt.Fprintf(w, "# TYPE neugo_feedback_total counter\n")
	fmt.Fprintf(w, "neugo_feedback_total %d\n", m.feedbackTotal.Load())

	fmt.Fprintf(w, "# TYPE neugo_feedback_dropped_total counter\n")
	fmt.Fprintf(w, "neugo_feedback_dropped_total %d\n", m.feedbackDropped.Load())

	fmt.Fprintf(w, "# TYPE neugo_swap_total counter\n")
	fmt.Fprintf(w, "neugo_swap_total %d\n", m.swapTotal.Load())

	fmt.Fprintf(w, "# TYPE neugo_swap_rejected_total counter\n")
	fmt.Fprintf(w, "neugo_swap_rejected_total %d\n", m.swapRejectedTotal.Load())

	fmt.Fprintf(w, "# TYPE neugo_model_generation gauge\n")
	fmt.Fprintf(w, "neugo_model_generation %d\n", m.modelGen.Load())
}
