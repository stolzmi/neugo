package serve

import (
	"bytes"
	"regexp"
	"testing"
)

func TestMetricsWritePrometheus(t *testing.T) {
	m := &metrics{}

	// Bump predictTotal twice
	m.predictTotal.Add(1)
	m.predictTotal.Add(1)

	// Set modelGen to 7
	m.modelGen.Store(7)

	// Capture writePrometheus output
	var buf bytes.Buffer
	m.writePrometheus(&buf)

	output := buf.String()

	// Assert it contains neugo_predict_total 2
	if !bytes.Contains(buf.Bytes(), []byte("neugo_predict_total 2")) {
		t.Errorf("output does not contain 'neugo_predict_total 2':\n%s", output)
	}

	// Assert it contains neugo_model_generation 7
	if !bytes.Contains(buf.Bytes(), []byte("neugo_model_generation 7")) {
		t.Errorf("output does not contain 'neugo_model_generation 7':\n%s", output)
	}

	// Assert every non-# line matches ^[a-z_]+ [0-9]+$
	lines := bytes.Split(buf.Bytes(), []byte("\n"))
	metricLineRegex := regexp.MustCompile(`^[a-z_]+ [0-9]+$`)
	for _, line := range lines {
		if len(line) == 0 {
			continue // Skip empty lines
		}
		if bytes.HasPrefix(line, []byte("#")) {
			continue // Skip TYPE/HELP lines
		}
		if !metricLineRegex.Match(line) {
			t.Errorf("metric line does not match pattern: %s", string(line))
		}
	}
}
