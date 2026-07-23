// text/dataset.go
package text

import (
	"fmt"
	"os"
	"strings"
)

// LoadLineDataset reads path and returns its non-empty lines — a plain
// line-delimited corpus loader (one training example per line), paired
// with TrainBPE/Encode for language-modeling-style datasets. Trailing
// '\r' is stripped from each line so CRLF-terminated files load the same
// as LF-terminated ones.
func LoadLineDataset(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("text: LoadLineDataset: %w", err)
	}
	var lines []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimRight(line, "\r")
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	return lines, nil
}
