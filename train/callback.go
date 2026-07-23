package train

import (
	"fmt"
	"math"
	"github.com/stolzmi/neugo/nn"
	"time"
)

// Callback observes training events. Hooks receive plain data (never a
// *Trainer reference) so callbacks can't accidentally hold onto shared,
// mutable trainer state across Fit calls — see Task 10 note in the plan.
type Callback interface {
	OnTrainBegin()
	OnTrainEnd()
	OnEpochBegin(epoch int)
	OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param)
	OnBatchEnd(batch int, loss float32)
}

// BaseCallback provides no-op defaults; embed it to implement only the
// hooks you need.
type BaseCallback struct{}

func (BaseCallback) OnTrainBegin()          {}
func (BaseCallback) OnTrainEnd()            {}
func (BaseCallback) OnEpochBegin(epoch int) {}
func (BaseCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
}
func (BaseCallback) OnBatchEnd(batch int, loss float32) {}

// History accumulates per-epoch losses/metrics. Trainer.Fit constructs and
// returns a fresh *History on every call — it is never something a caller
// hands into train.Callbacks(...) and never stored on the Trainer itself,
// which is what fixes the old "History accumulates across Train calls" bug
// by construction rather than by caller discipline.
type History struct {
	BaseCallback
	TrainLoss              []float32
	ValLoss, ValAcc, ValF1 []float32
	StartTime, EndTime     time.Time
}

func NewHistory() *History { return &History{} }

func (h *History) OnTrainBegin() { h.StartTime = time.Now() }
func (h *History) OnTrainEnd()   { h.EndTime = time.Now() }

func (h *History) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	h.TrainLoss = append(h.TrainLoss, trainLoss)
	if valMetrics != nil {
		h.ValLoss = append(h.ValLoss, valMetrics.Loss)
		h.ValAcc = append(h.ValAcc, valMetrics.Accuracy)
		h.ValF1 = append(h.ValF1, valMetrics.F1Score)
	}
}

func (h *History) Duration() time.Duration { return h.EndTime.Sub(h.StartTime) }

// EarlyStoppingCallback stops training after Patience epochs without a
// Loss (train loss, or validation loss when validation data is supplied)
// improvement of at least MinDelta, and can restore the best in-memory
// weight snapshot afterward.
type EarlyStoppingCallback struct {
	BaseCallback
	Patience   int
	MinDelta   float32
	ShouldStop bool

	bestLoss   float32
	counter    int
	bestParams [][]float32
}

func EarlyStopping(patience int) *EarlyStoppingCallback {
	return &EarlyStoppingCallback{Patience: patience, bestLoss: float32(math.Inf(1))}
}

func (es *EarlyStoppingCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	monitor := trainLoss
	if valMetrics != nil {
		monitor = valMetrics.Loss
	}
	if monitor < es.bestLoss-es.MinDelta {
		es.bestLoss = monitor
		es.counter = 0
		es.bestParams = snapshotParamValues(params)
	} else {
		es.counter++
		if es.counter >= es.Patience {
			es.ShouldStop = true
		}
	}
}

func (es *EarlyStoppingCallback) RestoreBestWeights(params []*nn.Param) {
	if es.bestParams == nil {
		return
	}
	for i, p := range params {
		copy(p.Value.Data, es.bestParams[i])
	}
}

func snapshotParamValues(params []*nn.Param) [][]float32 {
	snap := make([][]float32, len(params))
	for i, p := range params {
		snap[i] = append([]float32(nil), p.Value.Data...)
	}
	return snap
}

// ModelCheckpointCallback saves the model when Monitor improves (or every
// epoch, if SaveBestOnly is false). Save is left nil here — Trainer.Fit
// wires it to nn.Save(model, path) once nn.Save exists (Task 21); until
// then a ModelCheckpointCallback with a nil Save is a documented no-op.
// Failures are recorded in LastError rather than printed, keeping stdout
// output limited to ProgressBar/Summary per the Global Constraints.
type ModelCheckpointCallback struct {
	BaseCallback
	Filepath     string
	Monitor      string // "loss", "accuracy", "f1"
	Mode         string // "min" or "max"
	SaveBestOnly bool
	Save         func(path string) error
	LastError    error

	bestValue float32
}

func ModelCheckpoint(filepath, monitor, mode string, saveBestOnly bool) *ModelCheckpointCallback {
	best := float32(math.Inf(1))
	if mode == "max" {
		best = float32(math.Inf(-1))
	}
	return &ModelCheckpointCallback{Filepath: filepath, Monitor: monitor, Mode: mode, SaveBestOnly: saveBestOnly, bestValue: best}
}

func (mc *ModelCheckpointCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	if mc.Save == nil || valMetrics == nil {
		return
	}
	var current float32
	switch mc.Monitor {
	case "accuracy":
		current = valMetrics.Accuracy
	case "f1":
		current = valMetrics.F1Score
	default:
		current = valMetrics.Loss
	}
	improved := current < mc.bestValue
	if mc.Mode == "max" {
		improved = current > mc.bestValue
	}
	if improved {
		mc.bestValue = current
	}
	if improved || !mc.SaveBestOnly {
		mc.LastError = mc.Save(mc.Filepath)
	}
}

// ProgressBarCallback is one of the stdout-writing callbacks in this
// package (alongside TUICallback and GradientHistogramCallback; nn.Summary
// is the one stdout writer outside train).
type ProgressBarCallback struct {
	BaseCallback
	TotalEpochs int
	PrintEvery  int
}

func ProgressBar(totalEpochs, printEvery int) *ProgressBarCallback {
	return &ProgressBarCallback{TotalEpochs: totalEpochs, PrintEvery: printEvery}
}

func (pb *ProgressBarCallback) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	if pb.PrintEvery <= 0 || (epoch%pb.PrintEvery != 0 && epoch != pb.TotalEpochs-1) {
		return
	}
	if valMetrics != nil {
		fmt.Printf("Epoch %d/%d - loss: %.4f - val_loss: %.4f - val_acc: %.2f%%\n",
			epoch+1, pb.TotalEpochs, trainLoss, valMetrics.Loss, valMetrics.Accuracy)
	} else {
		fmt.Printf("Epoch %d/%d - loss: %.4f\n", epoch+1, pb.TotalEpochs, trainLoss)
	}
}

// CallbackList fans every hook out to each registered Callback in order.
type CallbackList struct {
	callbacks []Callback
}

func NewCallbackList(cbs ...Callback) *CallbackList { return &CallbackList{callbacks: cbs} }

func (cl *CallbackList) Add(cb Callback) { cl.callbacks = append(cl.callbacks, cb) }

func (cl *CallbackList) OnTrainBegin() {
	for _, cb := range cl.callbacks {
		cb.OnTrainBegin()
	}
}

func (cl *CallbackList) OnTrainEnd() {
	for _, cb := range cl.callbacks {
		cb.OnTrainEnd()
	}
}

func (cl *CallbackList) OnEpochBegin(epoch int) {
	for _, cb := range cl.callbacks {
		cb.OnEpochBegin(epoch)
	}
}

func (cl *CallbackList) OnEpochEnd(epoch int, trainLoss float32, valMetrics *Metrics, params []*nn.Param) {
	for _, cb := range cl.callbacks {
		cb.OnEpochEnd(epoch, trainLoss, valMetrics, params)
	}
}

func (cl *CallbackList) OnBatchEnd(batch int, loss float32) {
	for _, cb := range cl.callbacks {
		cb.OnBatchEnd(batch, loss)
	}
}
