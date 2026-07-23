// train/experimentlog.go
package train

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/stolzmi/neugo/nn"
)

// experimentLogEntry is one line of the JSONL file ExperimentLogCallback
// writes. Type distinguishes the three kinds of line a run produces:
// "run_start" (once, with Meta), "epoch" (once per epoch), and "run_end"
// (once, closing out the run) — all sharing one RunID so multiple runs
// appended to the same file (e.g. across a hyperparameter sweep) can be
// grouped back together later.
type experimentLogEntry struct {
	Type      string            `json:"type"`
	RunID     string            `json:"run_id"`
	Epoch     int               `json:"epoch,omitempty"`
	TrainLoss float32           `json:"train_loss,omitempty"`
	ValLoss   float32           `json:"val_loss,omitempty"`
	ValAcc    float32           `json:"val_accuracy,omitempty"`
	ValF1     float32           `json:"val_f1,omitempty"`
	Meta      map[string]string `json:"meta,omitempty"`
	Time      time.Time         `json:"time"`
}

// ExperimentLogCallback appends one JSON line per event (run start, each
// epoch, run end) to Path — a lightweight, dependency-free alternative to
// an external experiment tracker (wandb/MLflow/...) for comparing sweeps
// later: load the JSONL file, group by run_id, plot whatever you need.
// Opens Path in append mode so multiple training runs (e.g. every trial
// in a train.tune sweep) can share one log file. Write failures are
// captured in LastError rather than panicking, matching
// ModelCheckpointCallback's convention.
type ExperimentLogCallback struct {
	BaseCallback
	Path      string
	Meta      map[string]string
	RunID     string
	LastError error

	file *os.File
	enc  *json.Encoder
}

// ExperimentLog creates a callback appending JSONL run/epoch records to
// path. meta is written once as part of the "run_start" record — put
// hyperparameters, a dataset identifier, a git commit, whatever you want
// to be able to tell this run apart from another later.
func ExperimentLog(path string, meta map[string]string) *ExperimentLogCallback {
	return &ExperimentLogCallback{Path: path, Meta: meta}
}

func (c *ExperimentLogCallback) write(entry experimentLogEntry) {
	if c.LastError != nil || c.enc == nil {
		return
	}
	entry.RunID = c.RunID
	entry.Time = time.Now()
	if err := c.enc.Encode(entry); err != nil {
		c.LastError = fmt.Errorf("train: ExperimentLog: %w", err)
	}
}

func (c *ExperimentLogCallback) OnTrainBegin() {
	f, err := os.OpenFile(c.Path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		c.LastError = fmt.Errorf("train: ExperimentLog: %w", err)
		return
	}
	c.file = f
	c.enc = json.NewEncoder(f)
	c.RunID = fmt.Sprintf("run-%d", time.Now().UnixNano())
	c.write(experimentLogEntry{Type: "run_start", Meta: c.Meta})
}

func (c *ExperimentLogCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	entry := experimentLogEntry{Type: "epoch", Epoch: epoch, TrainLoss: trainLoss}
	if valMetrics != nil {
		entry.ValLoss = valMetrics.Loss
		entry.ValAcc = valMetrics.Accuracy
		entry.ValF1 = valMetrics.F1Score
	}
	c.write(entry)
}

func (c *ExperimentLogCallback) OnTrainEnd() {
	c.write(experimentLogEntry{Type: "run_end"})
	if c.file == nil {
		return
	}
	if err := c.file.Close(); err != nil && c.LastError == nil {
		c.LastError = fmt.Errorf("train: ExperimentLog: %w", err)
	}
}
