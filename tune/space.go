package tune

import (
	"fmt"
	"math"
	"math/rand"
)

// Space defines a search space with ordered parameter definitions.
// Order matters for deterministic sampling, so we use a slice, not a map.
type Space struct {
	params []paramDef
}

// paramDef represents a single parameter definition.
type paramDef interface {
	sample(r *rand.Rand) (string, any)
}

// floatParam defines a uniform float parameter.
type floatParam struct {
	name string
	min  float64
	max  float64
}

func (p *floatParam) sample(r *rand.Rand) (string, any) {
	v := p.min + r.Float64()*(p.max-p.min)
	return p.name, v
}

// logFloatParam defines a log-uniform float parameter.
type logFloatParam struct {
	name string
	min  float64
	max  float64
	lo   float64 // log(min)
	hi   float64 // log(max)
}

func (p *logFloatParam) sample(r *rand.Rand) (string, any) {
	v := math.Exp(p.lo + r.Float64()*(p.hi-p.lo))
	return p.name, v
}

// intParam defines an integer parameter (inclusive both ends).
type intParam struct {
	name string
	min  int
	max  int
}

func (p *intParam) sample(r *rand.Rand) (string, any) {
	v := p.min + r.Intn(p.max-p.min+1)
	return p.name, v
}

// choiceParam defines a categorical parameter.
type choiceParam struct {
	name    string
	options []string
}

func (p *choiceParam) sample(r *rand.Rand) (string, any) {
	idx := r.Intn(len(p.options))
	return p.name, p.options[idx]
}

// NewSpace creates a new search space.
func NewSpace() *Space {
	return &Space{
		params: []paramDef{},
	}
}

// Float adds a uniform float parameter.
func (s *Space) Float(name string, min, max float64) *Space {
	s.params = append(s.params, &floatParam{
		name: name,
		min:  min,
		max:  max,
	})
	return s
}

// LogFloat adds a log-uniform float parameter.
// Panics if min <= 0 or max < min.
func (s *Space) LogFloat(name string, min, max float64) *Space {
	if min <= 0 {
		panic(fmt.Sprintf("LogFloat(%s): min must be > 0, got %v", name, min))
	}
	if max < min {
		panic(fmt.Sprintf("LogFloat(%s): max must be >= min, got max=%v, min=%v", name, max, min))
	}
	s.params = append(s.params, &logFloatParam{
		name: name,
		min:  min,
		max:  max,
		lo:   math.Log(min),
		hi:   math.Log(max),
	})
	return s
}

// Int adds an integer parameter (inclusive both ends).
func (s *Space) Int(name string, min, max int) *Space {
	s.params = append(s.params, &intParam{
		name: name,
		min:  min,
		max:  max,
	})
	return s
}

// Choice adds a categorical parameter.
func (s *Space) Choice(name string, options ...string) *Space {
	s.params = append(s.params, &choiceParam{
		name:    name,
		options: options,
	})
	return s
}

// Sample generates a sample from the space using the provided random source.
func (s *Space) Sample(r *rand.Rand) Params {
	result := make(Params)
	for _, param := range s.params {
		name, value := param.sample(r)
		result[name] = value
	}
	return result
}

// Params is a map of parameter values.
type Params map[string]any

// Float retrieves a float parameter value.
// Panics with a clear message if the name is missing or the type is wrong.
func (p Params) Float(name string) float64 {
	v, ok := p[name]
	if !ok {
		panic(fmt.Sprintf("Params.Float(%q): parameter not found", name))
	}
	f, ok := v.(float64)
	if !ok {
		panic(fmt.Sprintf("Params.Float(%q): expected float64, got %T", name, v))
	}
	return f
}

// Int retrieves an int parameter value.
// Panics with a clear message if the name is missing or the type is wrong.
func (p Params) Int(name string) int {
	v, ok := p[name]
	if !ok {
		panic(fmt.Sprintf("Params.Int(%q): parameter not found", name))
	}
	i, ok := v.(int)
	if !ok {
		panic(fmt.Sprintf("Params.Int(%q): expected int, got %T", name, v))
	}
	return i
}

// Choice retrieves a string parameter value.
// Panics with a clear message if the name is missing or the type is wrong.
func (p Params) Choice(name string) string {
	v, ok := p[name]
	if !ok {
		panic(fmt.Sprintf("Params.Choice(%q): parameter not found", name))
	}
	s, ok := v.(string)
	if !ok {
		panic(fmt.Sprintf("Params.Choice(%q): expected string, got %T", name, v))
	}
	return s
}
