package tune

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestSpaceFloat(t *testing.T) {
	s := NewSpace().Float("x", -2, 2)
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 1000; i++ {
		params := s.Sample(r)
		v := params.Float("x")
		if v < -2 || v > 2 {
			t.Errorf("Float sample %d out of range: %v", i, v)
		}
	}
}

func TestSpaceLogFloat(t *testing.T) {
	s := NewSpace().LogFloat("lr", 1e-4, 1e-1)
	r := rand.New(rand.NewSource(1))

	var below1e3 int
	for i := 0; i < 1000; i++ {
		params := s.Sample(r)
		v := params.Float("lr")
		if v < 1e-4 || v > 1e-1 {
			t.Errorf("LogFloat sample %d out of range: %v", i, v)
		}
		if v < 1e-3 {
			below1e3++
		}
	}

	// At least 10% should be below 1e-3 (log-uniform)
	// Uniform would give ~0.9%
	pct := float64(below1e3) / 1000.0
	if pct < 0.10 {
		t.Errorf("LogFloat: only %.1f%% below 1e-3, want >=10%% (log-uniform)", pct*100)
	}
}

func TestSpaceInt(t *testing.T) {
	s := NewSpace().Int("h", 4, 8)
	r := rand.New(rand.NewSource(1))

	minSeen := 9
	maxSeen := 3
	for i := 0; i < 1000; i++ {
		params := s.Sample(r)
		v := params.Int("h")
		if v < 4 || v > 8 {
			t.Errorf("Int sample %d out of range: %v", i, v)
		}
		if v < minSeen {
			minSeen = v
		}
		if v > maxSeen {
			maxSeen = v
		}
	}

	if minSeen != 4 {
		t.Errorf("Int: min seen %d, want 4", minSeen)
	}
	if maxSeen != 8 {
		t.Errorf("Int: max seen %d, want 8", maxSeen)
	}
}

func TestSpaceChoice(t *testing.T) {
	s := NewSpace().Choice("opt", "a", "b", "c")
	r := rand.New(rand.NewSource(1))
	validMap := map[string]bool{"a": true, "b": true, "c": true}

	for i := 0; i < 1000; i++ {
		params := s.Sample(r)
		v := params.Choice("opt")
		if !validMap[v] {
			t.Errorf("Choice sample %d invalid: %v", i, v)
		}
	}
}

func TestSpaceDeterminism(t *testing.T) {
	s1 := NewSpace().Float("x", 0, 1).Int("y", 0, 10)
	s2 := NewSpace().Float("x", 0, 1).Int("y", 0, 10)

	r1 := rand.New(rand.NewSource(42))
	r2 := rand.New(rand.NewSource(42))

	for i := 0; i < 20; i++ {
		p1 := s1.Sample(r1)
		p2 := s2.Sample(r2)
		if !reflect.DeepEqual(p1, p2) {
			t.Errorf("Determinism failed at draw %d: %v != %v", i, p1, p2)
		}
	}
}

func TestParamsMissingName(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for missing name")
		}
	}()

	s := NewSpace().Float("x", 0, 1)
	r := rand.New(rand.NewSource(1))
	params := s.Sample(r)
	_ = params.Float("missing")
}

func TestParamsWrongKind(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for wrong kind")
		}
	}()

	s := NewSpace().Float("x", 0, 1)
	r := rand.New(rand.NewSource(1))
	params := s.Sample(r)
	_ = params.Int("x")
}

func TestLogFloatPanicMinLeZero(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for min <= 0")
		}
	}()

	NewSpace().LogFloat("x", 0, 1)
}

func TestLogFloatPanicMaxLessThanMin(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for max < min")
		}
	}()

	NewSpace().LogFloat("x", 1, 0.5)
}
