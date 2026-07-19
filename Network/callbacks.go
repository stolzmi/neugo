package Network

import (
	"fmt"
	"time"
)

// Callback interface for training events
type Callback interface {
	OnTrainBegin(network *Network)
	OnTrainEnd(network *Network)
	OnEpochBegin(epoch int, network *Network)
	OnEpochEnd(epoch int, network *Network, metrics *Metrics)
	OnBatchBegin(batch int, network *Network)
	OnBatchEnd(batch int, network *Network, loss float32)
}

// BaseCallback provides default no-op implementations
type BaseCallback struct{}

func (cb *BaseCallback) OnTrainBegin(network *Network)                              {}
func (cb *BaseCallback) OnTrainEnd(network *Network)                                {}
func (cb *BaseCallback) OnEpochBegin(epoch int, network *Network)                   {}
func (cb *BaseCallback) OnEpochEnd(epoch int, network *Network, metrics *Metrics)   {}
func (cb *BaseCallback) OnBatchBegin(batch int, network *Network)                   {}
func (cb *BaseCallback) OnBatchEnd(batch int, network *Network, loss float32)       {}

// History tracks training metrics over time
type History struct {
	BaseCallback
	TrainLoss []float32
	ValLoss   []float32
	ValAcc    []float32
	ValF1     []float32
	Epochs    []int
	StartTime time.Time
	EndTime   time.Time
}

func NewHistory() *History {
	return &History{
		TrainLoss: make([]float32, 0),
		ValLoss:   make([]float32, 0),
		ValAcc:    make([]float32, 0),
		ValF1:     make([]float32, 0),
		Epochs:    make([]int, 0),
	}
}

func (h *History) OnTrainBegin(network *Network) {
	h.StartTime = time.Now()
}

func (h *History) OnTrainEnd(network *Network) {
	h.EndTime = time.Now()
}

func (h *History) OnEpochEnd(epoch int, network *Network, metrics *Metrics) {
	h.Epochs = append(h.Epochs, epoch)
	if metrics != nil {
		h.ValLoss = append(h.ValLoss, metrics.Loss)
		h.ValAcc = append(h.ValAcc, metrics.Accuracy)
		h.ValF1 = append(h.ValF1, metrics.F1Score)
	}
}

func (h *History) RecordTrainLoss(loss float32) {
	h.TrainLoss = append(h.TrainLoss, loss)
}

func (h *History) Duration() time.Duration {
	return h.EndTime.Sub(h.StartTime)
}

// ModelCheckpoint saves the best model during training
type ModelCheckpoint struct {
	BaseCallback
	Filepath       string
	Monitor        string  // "loss", "accuracy", "f1"
	Mode           string  // "min" or "max"
	SaveBestOnly   bool
	BestValue      float32
	Verbose        bool
}

func NewModelCheckpoint(filepath string, monitor string, mode string, saveBestOnly bool, verbose bool) *ModelCheckpoint {
	bestValue := float32(1e9)
	if mode == "max" {
		bestValue = float32(-1e9)
	}

	return &ModelCheckpoint{
		Filepath:     filepath,
		Monitor:      monitor,
		Mode:         mode,
		SaveBestOnly: saveBestOnly,
		BestValue:    bestValue,
		Verbose:      verbose,
	}
}

func (mc *ModelCheckpoint) OnEpochEnd(epoch int, network *Network, metrics *Metrics) {
	if metrics == nil {
		return
	}

	var currentValue float32
	switch mc.Monitor {
	case "loss":
		currentValue = metrics.Loss
	case "accuracy":
		currentValue = metrics.Accuracy
	case "f1":
		currentValue = metrics.F1Score
	default:
		currentValue = metrics.Loss
	}

	shouldSave := false
	if mc.Mode == "min" {
		if currentValue < mc.BestValue {
			mc.BestValue = currentValue
			shouldSave = true
		}
	} else {
		if currentValue > mc.BestValue {
			mc.BestValue = currentValue
			shouldSave = true
		}
	}

	if shouldSave || !mc.SaveBestOnly {
		err := network.SaveToFile(mc.Filepath)
		if err != nil && mc.Verbose {
			fmt.Printf("Warning: Failed to save model: %v\n", err)
		} else if mc.Verbose {
			fmt.Printf("Epoch %d: %s improved to %.4f, saving model to %s\n",
				epoch, mc.Monitor, currentValue, mc.Filepath)
		}
	}
}

// ProgressBar displays training progress
type ProgressBar struct {
	BaseCallback
	TotalEpochs int
	Verbose     bool
	PrintEvery  int // Print every N epochs
}

func NewProgressBar(totalEpochs int, printEvery int, verbose bool) *ProgressBar {
	return &ProgressBar{
		TotalEpochs: totalEpochs,
		PrintEvery:  printEvery,
		Verbose:     verbose,
	}
}

func (pb *ProgressBar) OnTrainBegin(network *Network) {
	if pb.Verbose {
		fmt.Println("Training started...")
	}
}

func (pb *ProgressBar) OnEpochEnd(epoch int, network *Network, metrics *Metrics) {
	if pb.Verbose && (epoch%pb.PrintEvery == 0 || epoch == pb.TotalEpochs-1) {
		progress := float64(epoch+1) / float64(pb.TotalEpochs) * 100
		if metrics != nil {
			fmt.Printf("Epoch %d/%d (%.1f%%) - Loss: %.4f, Acc: %.2f%%, F1: %.4f\n",
				epoch+1, pb.TotalEpochs, progress, metrics.Loss, metrics.Accuracy, metrics.F1Score)
		} else {
			fmt.Printf("Epoch %d/%d (%.1f%%) completed\n",
				epoch+1, pb.TotalEpochs, progress)
		}
	}
}

func (pb *ProgressBar) OnTrainEnd(network *Network) {
	if pb.Verbose {
		fmt.Println("Training completed!")
	}
}

// LearningRateLogger logs learning rate changes
type LearningRateLogger struct {
	BaseCallback
	History []float32
	Verbose bool
}

func NewLearningRateLogger(verbose bool) *LearningRateLogger {
	return &LearningRateLogger{
		History: make([]float32, 0),
		Verbose: verbose,
	}
}

func (lrl *LearningRateLogger) LogLR(epoch int, lr float32) {
	lrl.History = append(lrl.History, lr)
	if lrl.Verbose && epoch%10 == 0 {
		fmt.Printf("Epoch %d: Learning rate = %.6f\n", epoch, lr)
	}
}

// CustomCallback allows users to define custom behavior
type CustomCallback struct {
	BaseCallback
	OnTrainBeginFunc  func(*Network)
	OnTrainEndFunc    func(*Network)
	OnEpochBeginFunc  func(int, *Network)
	OnEpochEndFunc    func(int, *Network, *Metrics)
	OnBatchBeginFunc  func(int, *Network)
	OnBatchEndFunc    func(int, *Network, float32)
}

func NewCustomCallback() *CustomCallback {
	return &CustomCallback{}
}

func (cc *CustomCallback) OnTrainBegin(network *Network) {
	if cc.OnTrainBeginFunc != nil {
		cc.OnTrainBeginFunc(network)
	}
}

func (cc *CustomCallback) OnTrainEnd(network *Network) {
	if cc.OnTrainEndFunc != nil {
		cc.OnTrainEndFunc(network)
	}
}

func (cc *CustomCallback) OnEpochBegin(epoch int, network *Network) {
	if cc.OnEpochBeginFunc != nil {
		cc.OnEpochBeginFunc(epoch, network)
	}
}

func (cc *CustomCallback) OnEpochEnd(epoch int, network *Network, metrics *Metrics) {
	if cc.OnEpochEndFunc != nil {
		cc.OnEpochEndFunc(epoch, network, metrics)
	}
}

func (cc *CustomCallback) OnBatchBegin(batch int, network *Network) {
	if cc.OnBatchBeginFunc != nil {
		cc.OnBatchBeginFunc(batch, network)
	}
}

func (cc *CustomCallback) OnBatchEnd(batch int, network *Network, loss float32) {
	if cc.OnBatchEndFunc != nil {
		cc.OnBatchEndFunc(batch, network, loss)
	}
}

// CallbackList manages multiple callbacks
type CallbackList struct {
	Callbacks []Callback
}

func NewCallbackList(callbacks ...Callback) *CallbackList {
	return &CallbackList{Callbacks: callbacks}
}

func (cl *CallbackList) Add(callback Callback) {
	cl.Callbacks = append(cl.Callbacks, callback)
}

func (cl *CallbackList) OnTrainBegin(network *Network) {
	for _, cb := range cl.Callbacks {
		cb.OnTrainBegin(network)
	}
}

func (cl *CallbackList) OnTrainEnd(network *Network) {
	for _, cb := range cl.Callbacks {
		cb.OnTrainEnd(network)
	}
}

func (cl *CallbackList) OnEpochBegin(epoch int, network *Network) {
	for _, cb := range cl.Callbacks {
		cb.OnEpochBegin(epoch, network)
	}
}

func (cl *CallbackList) OnEpochEnd(epoch int, network *Network, metrics *Metrics) {
	for _, cb := range cl.Callbacks {
		cb.OnEpochEnd(epoch, network, metrics)
	}
}

func (cl *CallbackList) OnBatchBegin(batch int, network *Network) {
	for _, cb := range cl.Callbacks {
		cb.OnBatchBegin(batch, network)
	}
}

func (cl *CallbackList) OnBatchEnd(batch int, network *Network, loss float32) {
	for _, cb := range cl.Callbacks {
		cb.OnBatchEnd(batch, network, loss)
	}
}
