// cmd/neugo/diff.go
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// moduleDoc/paramDoc mirror nn/serialize.go's private JSON schema by JSON
// tag, not by importing nn — Save's output is just a JSON document, and
// this CLI only needs to walk its tree and compare numbers, not
// reconstruct real Module objects, so it decodes the file directly rather
// than asking the nn package to export its internal tree representation.
type moduleDoc struct {
	Type     string              `json:"type"`
	Params   map[string]paramDoc `json:"params,omitempty"`
	Modules  []moduleDoc         `json:"modules,omitempty"`
	Shortcut *moduleDoc          `json:"shortcut,omitempty"`
}

type paramDoc struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

// loadModuleDoc reads a file written by nn.Save or nn.SaveWithMetadata —
// the two formats differ only in whether the moduleDoc is the JSON root
// or nested under a "model" key.
func loadModuleDoc(path string) (moduleDoc, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return moduleDoc{}, err
	}
	var doc moduleDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		return moduleDoc{}, err
	}
	if doc.Type != "" {
		return doc, nil
	}
	var wrapper struct {
		Model moduleDoc `json:"model"`
	}
	if err := json.Unmarshal(data, &wrapper); err == nil && wrapper.Model.Type != "" {
		return wrapper.Model, nil
	}
	return moduleDoc{}, fmt.Errorf("%s does not look like an nn.Save/nn.SaveWithMetadata file (no \"type\" field found at the root or under \"model\")", path)
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// l2Diff returns the L2 norm of the elementwise difference between two
// equal-length float32 slices.
func l2Diff(a, b []float32) float64 {
	var sumSq float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sumSq += d * d
	}
	return math.Sqrt(sumSq)
}

// diffResult accumulates a report as it walks the tree, plus a running
// total weight-delta norm across every matched param — a single number
// answering "how different are these two checkpoints overall", with the
// full per-layer breakdown available in Report for anyone who needs to
// know exactly where.
type diffResult struct {
	Report        strings.Builder
	TotalDeltaSq  float64
	StructuralGap bool
}

func (d *diffResult) walk(a, c moduleDoc, path string) {
	if a.Type != c.Type {
		fmt.Fprintf(&d.Report, "%s: TYPE CHANGED %q -> %q\n", path, a.Type, c.Type)
		d.StructuralGap = true
		return
	}
	fmt.Fprintf(&d.Report, "%s: %s\n", path, a.Type)

	for name, pa := range a.Params {
		pc, ok := c.Params[name]
		if !ok {
			fmt.Fprintf(&d.Report, "  param %q: present in A, missing in B\n", name)
			d.StructuralGap = true
			continue
		}
		if !equalIntSlices(pa.Shape, pc.Shape) {
			fmt.Fprintf(&d.Report, "  param %q: SHAPE CHANGED %v -> %v\n", name, pa.Shape, pc.Shape)
			d.StructuralGap = true
			continue
		}
		norm := l2Diff(pa.Data, pc.Data)
		d.TotalDeltaSq += norm * norm
		fmt.Fprintf(&d.Report, "  param %q: |delta| = %.6f  (%d values)\n", name, norm, len(pa.Data))
	}
	for name := range c.Params {
		if _, ok := a.Params[name]; !ok {
			fmt.Fprintf(&d.Report, "  param %q: present in B, missing in A\n", name)
			d.StructuralGap = true
		}
	}

	n := len(a.Modules)
	if len(c.Modules) > n {
		n = len(c.Modules)
	}
	for i := 0; i < n; i++ {
		childPath := fmt.Sprintf("%s.modules[%d]", path, i)
		switch {
		case i >= len(a.Modules):
			fmt.Fprintf(&d.Report, "%s: present only in B (%s)\n", childPath, c.Modules[i].Type)
			d.StructuralGap = true
		case i >= len(c.Modules):
			fmt.Fprintf(&d.Report, "%s: present only in A (%s)\n", childPath, a.Modules[i].Type)
			d.StructuralGap = true
		default:
			d.walk(a.Modules[i], c.Modules[i], childPath)
		}
	}

	switch {
	case a.Shortcut == nil && c.Shortcut == nil:
	case a.Shortcut == nil:
		fmt.Fprintf(&d.Report, "%s.shortcut: present only in B\n", path)
		d.StructuralGap = true
	case c.Shortcut == nil:
		fmt.Fprintf(&d.Report, "%s.shortcut: present only in A\n", path)
		d.StructuralGap = true
	default:
		d.walk(*a.Shortcut, *c.Shortcut, path+".shortcut")
	}
}

// diffModels compares two nn.Save/nn.SaveWithMetadata files: architecture
// (module type at each position in the tree) and, for every matching
// leaf's params, the L2 norm of the weight delta — useful for "did my
// fine-tuning actually change this layer" or reviewing a PR that touches
// a checked-in model file.
func diffModels(pathA, pathB string) (string, error) {
	docA, err := loadModuleDoc(pathA)
	if err != nil {
		return "", fmt.Errorf("reading %s: %w", pathA, err)
	}
	docB, err := loadModuleDoc(pathB)
	if err != nil {
		return "", fmt.Errorf("reading %s: %w", pathB, err)
	}

	var d diffResult
	d.walk(docA, docB, "root")

	var out strings.Builder
	if d.StructuralGap {
		fmt.Fprintln(&out, "architecture: DIFFERS (see below)")
	} else {
		fmt.Fprintf(&out, "architecture: identical\ntotal weight delta |Δ|: %.6f\n", math.Sqrt(d.TotalDeltaSq))
	}
	fmt.Fprintln(&out, "---")
	out.WriteString(d.Report.String())
	return out.String(), nil
}

func runDiff(args []string, stdout io.Writer, stderr io.Writer) error {
	fs := flag.NewFlagSet("diff", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var pathA, pathB string
	fs.StringVar(&pathA, "a", "", "path to the first model JSON file")
	fs.StringVar(&pathB, "b", "", "path to the second model JSON file")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if pathA == "" || pathB == "" {
		return fmt.Errorf("-a and -b flags are both required")
	}

	report, err := diffModels(pathA, pathB)
	if err != nil {
		return err
	}
	fmt.Fprint(stdout, report)
	return nil
}
