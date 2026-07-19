package nn

import (
	"strings"
	"testing"
)

func TestParamCountMatchesSumOfParams(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 4},
		Linear(rng, 4, 5, XavierInit()), // 4*5 + 5 = 25
		ReLU(),
		Linear(rng, 5, 3, XavierInit()), // 5*3 + 3 = 18
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	if got, want := ParamCount(model), 43; got != want {
		t.Fatalf("ParamCount = %d, want %d", got, want)
	}
}

func TestSummaryListsEachLayerAndTotal(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 4},
		Linear(rng, 4, 5, XavierInit()),
		ReLU(),
		Linear(rng, 5, 3, XavierInit()),
		Softmax(),
	)
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	out, err := Summary(model, []int{2, 4})
	if err != nil {
		t.Fatalf("Summary: %v", err)
	}
	for _, want := range []string{"Linear", "ReLU", "Softmax", "Total params: 43"} {
		if !strings.Contains(out, want) {
			t.Errorf("Summary output missing %q:\n%s", want, out)
		}
	}
}

func TestSummaryPropagatesShapeError(t *testing.T) {
	rng := NewRNG(1)
	model, err := Sequential([]int{2, 4}, Linear(rng, 4, 5, XavierInit()))
	if err != nil {
		t.Fatalf("Sequential: %v", err)
	}
	if _, err := Summary(model, []int{2, 999}); err == nil {
		t.Fatal("expected error for a Summary inputShape that mismatches the built model, got nil")
	}
}
