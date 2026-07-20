package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/stolzmi/neugo/export"
)

func run(args []string, stderr io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("no subcommand provided")
	}

	subcommand := args[0]

	switch subcommand {
	case "export":
		return runExport(args[1:], stderr)
	default:
		return fmt.Errorf("unknown subcommand: %s", subcommand)
	}
}

func runExport(args []string, stderr io.Writer) error {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		modelPath string
		outPath   string
		pkg       string
		fn        string
	)

	fs.StringVar(&modelPath, "model", "", "path to model JSON file")
	fs.StringVar(&outPath, "out", "", "path to output Go file")
	fs.StringVar(&pkg, "pkg", "model", "package name")
	fs.StringVar(&fn, "fn", "Predict", "function name")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if modelPath == "" {
		return fmt.Errorf("-model flag is required")
	}

	if outPath == "" {
		return fmt.Errorf("-out flag is required")
	}

	// Read model JSON
	modelJSON, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("failed to read model file: %w", err)
	}

	// Generate Go code
	genCode, err := export.GenerateGo(modelJSON, export.Options{
		Package:  pkg,
		FuncName: fn,
	})
	if err != nil {
		return fmt.Errorf("failed to generate code: %w", err)
	}

	// Write output file
	if err := os.WriteFile(outPath, genCode, 0644); err != nil {
		return fmt.Errorf("failed to write output file: %w", err)
	}

	// Print success message to stderr
	fmt.Fprintf(stderr, "exported model to %s\n", outPath)

	return nil
}

func main() {
	err := run(os.Args[1:], os.Stderr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
