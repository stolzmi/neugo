// cmd/neugo/new.go
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// newTemplates holds one complete, working main.go body per architecture
// — "instant hello world" scaffolding for a new use case, the same way
// `cargo new` or `npm create` save you the blank-page cost of wiring up
// imports/Sequential/Trainer from scratch. {{PKG}} is the only
// substitution; everything else is real, working code that compiles
// as-is against this module (verified by TestNewSubcommandGeneratesBuildableProject).
var newTemplates = map[string]string{
	"mlp": `package {{PKG}}

import (
	"fmt"

	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

func main() {
	rng := nn.NewRNG(1)
	model, err := nn.Sequential([]int{4, 2},
		nn.Linear(rng, 2, 16, nn.HeInit()),
		nn.ReLU(),
		nn.Linear(rng, 16, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}
	summary, _ := nn.Summary(model, []int{4, 2})
	fmt.Print(summary)

	// Replace with your own data.
	x, _ := nn.NewTensorFromData([]float32{0, 0, 0, 1, 1, 0, 1, 1}, []int{4, 2})
	y, _ := nn.NewTensorFromData([]float32{0, 1, 1, 0}, []int{4, 1})

	trainer := train.New(model, train.Adam(0.05, 0.9, 0.999, 1e-8), train.BCELoss())
	hist, err := trainer.Fit(x, y, train.Epochs(1000), train.BatchSize(4), train.Seed(1))
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("final train loss: %.4f\n", hist.TrainLoss[len(hist.TrainLoss)-1])
}
`,
	"cnn": `package {{PKG}}

import (
	"fmt"

	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

func main() {
	rng := nn.NewRNG(1)
	const batch, h, w, c = 8, 16, 16, 1
	model, err := nn.Sequential([]int{batch, h, w, c},
		nn.Conv2DSame(rng, c, 8, 3, nn.HeInit()),
		nn.BatchNorm(8),
		nn.ReLU(),
		nn.MaxPool2D(2, 2),
		nn.Flatten(),
		nn.Linear(rng, 0, 32, nn.HeInit()), // 0 == infer from Flatten's output
		nn.ReLU(),
		nn.Linear(rng, 32, 10, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}
	summary, _ := nn.Summary(model, []int{batch, h, w, c})
	fmt.Print(summary)

	// Replace with real image data (see the data package's image loaders,
	// or data.LoadImageFolder for an ImageNet-style directory layout).
	x := nn.NewTensor([]int{batch, h, w, c})
	y := nn.NewTensor([]int{batch, 10})
	for i := 0; i < batch; i++ {
		y.Data[i*10+i%10] = 1
	}

	trainer := train.New(model, train.Adam(0.001, 0.9, 0.999, 1e-8), train.CrossEntropy())
	hist, err := trainer.Fit(x, y, train.Epochs(50), train.BatchSize(4), train.Seed(1))
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("final train loss: %.4f\n", hist.TrainLoss[len(hist.TrainLoss)-1])
}
`,
	"transformer": `package {{PKG}}

import (
	"fmt"

	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

func main() {
	const vocabSize, seqLen, dModel, numHeads, ffHidden = 100, 8, 32, 4, 64
	rng := nn.NewRNG(1)

	block, err := nn.TransformerBlock(rng, dModel, numHeads, ffHidden, true, nn.XavierInit())
	if err != nil {
		fmt.Println("build TransformerBlock:", err)
		return
	}
	model, err := nn.Sequential([]int{1, seqLen},
		nn.Embedding(rng, vocabSize, dModel, nn.NormalInit(0, 0.5)),
		nn.PositionalEmbedding(rng, seqLen, dModel, nn.NormalInit(0, 0.1)),
		block,
		nn.Flatten(),
		nn.Linear(rng, 0, vocabSize, nn.XavierInit()),
		nn.Softmax(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}
	summary, _ := nn.Summary(model, []int{1, seqLen})
	fmt.Print(summary)

	// Replace with real tokenized sequences (see the text package's
	// TrainBPE/Encode for a byte-level BPE tokenizer).
	x := nn.NewTensor([]int{1, seqLen})
	y := nn.NewTensor([]int{1, vocabSize})
	y.Data[3] = 1

	trainer := train.New(model, train.Adam(0.001, 0.9, 0.999, 1e-8), train.CrossEntropy())
	hist, err := trainer.Fit(x, y, train.Epochs(50), train.BatchSize(1), train.Seed(1))
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("final train loss: %.4f\n", hist.TrainLoss[len(hist.TrainLoss)-1])
}
`,
	"rnn": `package {{PKG}}

import (
	"fmt"

	"github.com/stolzmi/neugo/nn"
	"github.com/stolzmi/neugo/train"
)

func main() {
	const vocabSize, seqLen, dModel, hidden = 20, 6, 16, 32
	rng := nn.NewRNG(1)

	model, err := nn.Sequential([]int{1, seqLen},
		nn.Embedding(rng, vocabSize, dModel, nn.NormalInit(0, 0.5)),
		nn.LSTM(rng, dModel, hidden, nn.XavierInit()),
		nn.LastTimestep(),
		nn.Linear(rng, hidden, 1, nn.XavierInit()),
		nn.Sigmoid(),
	)
	if err != nil {
		fmt.Println("build model:", err)
		return
	}
	summary, _ := nn.Summary(model, []int{1, seqLen})
	fmt.Print(summary)

	// Replace with real sequence data (token ids as float32).
	x := nn.NewTensor([]int{1, seqLen})
	y, _ := nn.NewTensorFromData([]float32{1}, []int{1, 1})

	trainer := train.New(model, train.Adam(0.01, 0.9, 0.999, 1e-8), train.BCELoss())
	hist, err := trainer.Fit(x, y, train.Epochs(50), train.BatchSize(1), train.Seed(1))
	if err != nil {
		fmt.Println("fit:", err)
		return
	}
	fmt.Printf("final train loss: %.4f\n", hist.TrainLoss[len(hist.TrainLoss)-1])
}
`,
}

func newArchNames() []string {
	names := make([]string, 0, len(newTemplates))
	for name := range newTemplates {
		names = append(names, name)
	}
	return names
}

func runNew(args []string, stderr io.Writer) error {
	fs := flag.NewFlagSet("new", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var arch, out, pkg string
	fs.StringVar(&arch, "arch", "", fmt.Sprintf("architecture to scaffold (%s)", strings.Join(newArchNames(), ", ")))
	fs.StringVar(&out, "out", "", "output directory (created if it doesn't exist)")
	fs.StringVar(&pkg, "pkg", "main", "package name for the generated file")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if out == "" {
		return fmt.Errorf("-out flag is required")
	}
	tmpl, ok := newTemplates[arch]
	if !ok {
		return fmt.Errorf("unknown -arch %q, want one of: %s", arch, strings.Join(newArchNames(), ", "))
	}

	if err := os.MkdirAll(out, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	src := strings.ReplaceAll(tmpl, "{{PKG}}", pkg)
	outPath := filepath.Join(out, "main.go")
	if err := os.WriteFile(outPath, []byte(src), 0644); err != nil {
		return fmt.Errorf("failed to write %s: %w", outPath, err)
	}

	fmt.Fprintf(stderr, "scaffolded %s architecture at %s\n", arch, outPath)
	return nil
}
