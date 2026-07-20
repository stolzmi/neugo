package nn

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
)

type paramDoc struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

func toParamDoc(p *Param) paramDoc { return paramDoc{Shape: p.Value.Shape, Data: p.Value.Data} }

type moduleDoc struct {
	Type    string              `json:"type"`
	Config  json.RawMessage     `json:"config,omitempty"`
	Params  map[string]paramDoc `json:"params,omitempty"`
	Modules []moduleDoc         `json:"modules,omitempty"`
	// Shortcut is only set for "residual" nodes: nil means an identity
	// shortcut, present means a projection module (see ResidualBlock).
	Shortcut *moduleDoc `json:"shortcut,omitempty"`
}

type linearConfig struct {
	InFeatures  int `json:"in_features"`
	OutFeatures int `json:"out_features"`
}

type conv2DConfig struct {
	InChannels  int `json:"in_channels"`
	OutChannels int `json:"out_channels"`
	KernelSize  int `json:"kernel_size"`
	Padding     int `json:"padding"`
	Stride      int `json:"stride"`
}

type poolConfig struct {
	PoolSize int `json:"pool_size"`
	Stride   int `json:"stride"`
}

type dropoutConfig struct {
	Rate float32 `json:"rate"`
}

type batchNormConfig struct {
	Channels    int       `json:"channels"`
	RunningMean []float32 `json:"running_mean"`
	RunningVar  []float32 `json:"running_var"`
}

type groupNormConfig struct {
	Groups   int `json:"groups"`
	Channels int `json:"channels"`
}

type embeddingConfig struct {
	VocabSize int `json:"vocab_size"`
	EmbedDim  int `json:"embed_dim"`
}

type conv1DConfig struct {
	InChannels  int `json:"in_channels"`
	OutChannels int `json:"out_channels"`
	KernelSize  int `json:"kernel_size"`
	Padding     int `json:"padding"`
	Stride      int `json:"stride"`
}

type convTranspose2DConfig struct {
	InChannels  int `json:"in_channels"`
	OutChannels int `json:"out_channels"`
	KernelSize  int `json:"kernel_size"`
	Padding     int `json:"padding"`
	Stride      int `json:"stride"`
}

type leakyReLUConfig struct {
	Alpha float32 `json:"alpha"`
}

func mustMarshal(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

func encodeModule(m Module) (moduleDoc, error) {
	switch v := m.(type) {
	case *LinearLayer:
		return moduleDoc{
			Type:   "linear",
			Config: mustMarshal(linearConfig{InFeatures: v.inFeatures, OutFeatures: v.outFeatures}),
			Params: map[string]paramDoc{"W": toParamDoc(v.W), "B": toParamDoc(v.B)},
		}, nil
	case *Conv2DLayer:
		return moduleDoc{
			Type:   "conv2d",
			Config: mustMarshal(conv2DConfig{InChannels: v.inChannels, OutChannels: v.outChannels, KernelSize: v.kernelSize, Padding: v.padding, Stride: v.stride}),
			Params: map[string]paramDoc{"W": toParamDoc(v.W), "B": toParamDoc(v.B)},
		}, nil
	case *MaxPool2DLayer:
		return moduleDoc{Type: "maxpool2d", Config: mustMarshal(poolConfig{PoolSize: v.poolSize, Stride: v.stride})}, nil
	case *AvgPool2DLayer:
		return moduleDoc{Type: "avgpool2d", Config: mustMarshal(poolConfig{PoolSize: v.poolSize, Stride: v.stride})}, nil
	case *FlattenLayer:
		return moduleDoc{Type: "flatten"}, nil
	case *DropoutLayer:
		return moduleDoc{Type: "dropout", Config: mustMarshal(dropoutConfig{Rate: v.rate})}, nil
	case *BatchNormLayer:
		return moduleDoc{
			Type:   "batchnorm",
			Config: mustMarshal(batchNormConfig{Channels: v.channels, RunningMean: v.runningMean, RunningVar: v.runningVar}),
			Params: map[string]paramDoc{"gamma": toParamDoc(v.Gamma), "beta": toParamDoc(v.Beta)},
		}, nil
	case *GroupNormLayer:
		return moduleDoc{
			Type:   "groupnorm",
			Config: mustMarshal(groupNormConfig{Groups: v.groups, Channels: v.channels}),
			Params: map[string]paramDoc{"gamma": toParamDoc(v.Gamma), "beta": toParamDoc(v.Beta)},
		}, nil
	case *EmbeddingLayer:
		return moduleDoc{
			Type:   "embedding",
			Config: mustMarshal(embeddingConfig{VocabSize: v.vocabSize, EmbedDim: v.embedDim}),
			Params: map[string]paramDoc{"W": toParamDoc(v.W)},
		}, nil
	case *Conv1DLayer:
		return moduleDoc{
			Type:   "conv1d",
			Config: mustMarshal(conv1DConfig{InChannels: v.inChannels, OutChannels: v.outChannels, KernelSize: v.kernelSize, Padding: v.padding, Stride: v.stride}),
			Params: map[string]paramDoc{"W": toParamDoc(v.W), "B": toParamDoc(v.B)},
		}, nil
	case *ConvTranspose2DLayer:
		return moduleDoc{
			Type:   "convtranspose2d",
			Config: mustMarshal(convTranspose2DConfig{InChannels: v.inChannels, OutChannels: v.outChannels, KernelSize: v.kernelSize, Padding: v.padding, Stride: v.stride}),
			Params: map[string]paramDoc{"W": toParamDoc(v.W), "B": toParamDoc(v.B)},
		}, nil
	case *FrozenModule:
		innerDoc, err := encodeModule(v.inner)
		if err != nil {
			return moduleDoc{}, err
		}
		return moduleDoc{Type: "frozen", Modules: []moduleDoc{innerDoc}}, nil
	case *SoftmaxModule:
		return moduleDoc{Type: "softmax"}, nil
	case *ActivationModule:
		if v.Name() == "leaky_relu" {
			return moduleDoc{Type: "leaky_relu", Config: mustMarshal(leakyReLUConfig{Alpha: v.Alpha()})}, nil
		}
		return moduleDoc{Type: v.Name()}, nil
	case *SequentialModel:
		children := make([]moduleDoc, len(v.modules))
		for i, cm := range v.modules {
			cd, err := encodeModule(cm)
			if err != nil {
				return moduleDoc{}, err
			}
			children[i] = cd
		}
		return moduleDoc{Type: "sequential", Modules: children}, nil
	case *ResidualBlock:
		children := make([]moduleDoc, len(v.inner))
		for i, cm := range v.inner {
			cd, err := encodeModule(cm)
			if err != nil {
				return moduleDoc{}, err
			}
			children[i] = cd
		}
		doc := moduleDoc{Type: "residual", Modules: children}
		if v.shortcut != nil {
			sd, err := encodeModule(v.shortcut)
			if err != nil {
				return moduleDoc{}, err
			}
			doc.Shortcut = &sd
		}
		return doc, nil
	default:
		return moduleDoc{}, fmt.Errorf("nn: Save: unsupported module type %T", m)
	}
}

// encodeRoot validates model and encodes it to a moduleDoc.
func encodeRoot(model *SequentialModel, prefix string) (moduleDoc, error) {
	if model == nil {
		return moduleDoc{}, fmt.Errorf("nn: %s: model is nil", prefix)
	}
	return encodeModule(model)
}

// Save writes model as a JSON document — a tree of {type, config, params,
// modules} nodes readable by Load. RNG seed and optimizer state are never
// included (training-resume is out of scope).
func Save(model *SequentialModel, path string) error {
	doc, err := encodeRoot(model, "Save")
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return fmt.Errorf("nn: Save: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("nn: Save: %w", err)
	}
	return nil
}

func paramOrErr(doc moduleDoc, key, moduleType string) (paramDoc, error) {
	p, ok := doc.Params[key]
	if !ok {
		return paramDoc{}, fmt.Errorf("nn: Load: %s module missing %q param", moduleType, key)
	}
	return p, nil
}

// checkParamLen verifies a decoded paramDoc's data length matches the
// freshly-constructed target's length before it is copied in. Without this,
// copy(dst, src) silently truncates or zero-pads on a length mismatch,
// turning a corrupt or truncated JSON file into a successful Load with
// wrong weights instead of an error.
func checkParamLen(moduleType, key string, want int, p paramDoc) error {
	if len(p.Data) != want {
		return fmt.Errorf("nn: Load: %s module %q param has %d values, want %d", moduleType, key, len(p.Data), want)
	}
	return nil
}

func decodeModule(doc moduleDoc, rng *rand.Rand) (Module, error) {
	switch doc.Type {
	case "linear":
		var cfg linearConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: linear config: %w", err)
		}
		if cfg.InFeatures <= 0 || cfg.OutFeatures <= 0 {
			return nil, fmt.Errorf("nn: Load: linear config: in_features and out_features must be positive, got %d and %d", cfg.InFeatures, cfg.OutFeatures)
		}
		l := Linear(rng, cfg.InFeatures, cfg.OutFeatures, ZerosInit())
		w, err := paramOrErr(doc, "W", "linear")
		if err != nil {
			return nil, err
		}
		b, err := paramOrErr(doc, "B", "linear")
		if err != nil {
			return nil, err
		}
		if err := checkParamLen("linear", "W", len(l.W.Value.Data), w); err != nil {
			return nil, err
		}
		if err := checkParamLen("linear", "B", len(l.B.Value.Data), b); err != nil {
			return nil, err
		}
		copy(l.W.Value.Data, w.Data)
		copy(l.B.Value.Data, b.Data)
		return l, nil

	case "conv2d":
		var cfg conv2DConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: conv2d config: %w", err)
		}
		if cfg.InChannels <= 0 || cfg.OutChannels <= 0 || cfg.KernelSize <= 0 {
			return nil, fmt.Errorf("nn: Load: conv2d config: in_channels, out_channels, and kernel_size must be positive, got %d, %d, and %d", cfg.InChannels, cfg.OutChannels, cfg.KernelSize)
		}
		// Models saved before Stride existed have it zero-valued in JSON;
		// they were always stride-1, so default it here rather than fail.
		if cfg.Stride == 0 {
			cfg.Stride = 1
		}
		c := newConv2D(rng, cfg.InChannels, cfg.OutChannels, cfg.KernelSize, cfg.Padding, cfg.Stride, ZerosInit())
		w, err := paramOrErr(doc, "W", "conv2d")
		if err != nil {
			return nil, err
		}
		b, err := paramOrErr(doc, "B", "conv2d")
		if err != nil {
			return nil, err
		}
		if err := checkParamLen("conv2d", "W", len(c.W.Value.Data), w); err != nil {
			return nil, err
		}
		if err := checkParamLen("conv2d", "B", len(c.B.Value.Data), b); err != nil {
			return nil, err
		}
		copy(c.W.Value.Data, w.Data)
		copy(c.B.Value.Data, b.Data)
		return c, nil

	case "maxpool2d":
		var cfg poolConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: maxpool2d config: %w", err)
		}
		return MaxPool2D(cfg.PoolSize, cfg.Stride), nil

	case "avgpool2d":
		var cfg poolConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: avgpool2d config: %w", err)
		}
		return AvgPool2D(cfg.PoolSize, cfg.Stride), nil

	case "flatten":
		return Flatten(), nil

	case "dropout":
		var cfg dropoutConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: dropout config: %w", err)
		}
		return Dropout(cfg.Rate), nil

	case "batchnorm":
		var cfg batchNormConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: batchnorm config: %w", err)
		}
		if len(cfg.RunningMean) != cfg.Channels {
			return nil, fmt.Errorf("nn: Load: batchnorm module has %d running_mean values, want %d", len(cfg.RunningMean), cfg.Channels)
		}
		if len(cfg.RunningVar) != cfg.Channels {
			return nil, fmt.Errorf("nn: Load: batchnorm module has %d running_var values, want %d", len(cfg.RunningVar), cfg.Channels)
		}
		bn := BatchNorm(cfg.Channels)
		copy(bn.runningMean, cfg.RunningMean)
		copy(bn.runningVar, cfg.RunningVar)
		g, err := paramOrErr(doc, "gamma", "batchnorm")
		if err != nil {
			return nil, err
		}
		beta, err := paramOrErr(doc, "beta", "batchnorm")
		if err != nil {
			return nil, err
		}
		if err := checkParamLen("batchnorm", "gamma", len(bn.Gamma.Value.Data), g); err != nil {
			return nil, err
		}
		if err := checkParamLen("batchnorm", "beta", len(bn.Beta.Value.Data), beta); err != nil {
			return nil, err
		}
		copy(bn.Gamma.Value.Data, g.Data)
		copy(bn.Beta.Value.Data, beta.Data)
		return bn, nil

	case "groupnorm":
		var cfg groupNormConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: groupnorm config: %w", err)
		}
		if cfg.Groups <= 0 || cfg.Channels <= 0 || cfg.Channels%cfg.Groups != 0 {
			return nil, fmt.Errorf("nn: Load: groupnorm config: channels %d must be positive and evenly divisible by groups %d", cfg.Channels, cfg.Groups)
		}
		gn := GroupNorm(cfg.Groups, cfg.Channels)
		gamma, err := paramOrErr(doc, "gamma", "groupnorm")
		if err != nil {
			return nil, err
		}
		beta, err := paramOrErr(doc, "beta", "groupnorm")
		if err != nil {
			return nil, err
		}
		if err := checkParamLen("groupnorm", "gamma", len(gn.Gamma.Value.Data), gamma); err != nil {
			return nil, err
		}
		if err := checkParamLen("groupnorm", "beta", len(gn.Beta.Value.Data), beta); err != nil {
			return nil, err
		}
		copy(gn.Gamma.Value.Data, gamma.Data)
		copy(gn.Beta.Value.Data, beta.Data)
		return gn, nil

	case "embedding":
		var cfg embeddingConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: embedding config: %w", err)
		}
		if cfg.VocabSize <= 0 || cfg.EmbedDim <= 0 {
			return nil, fmt.Errorf("nn: Load: embedding config: vocab_size and embed_dim must be positive, got %d and %d", cfg.VocabSize, cfg.EmbedDim)
		}
		e := Embedding(rng, cfg.VocabSize, cfg.EmbedDim, ZerosInit())
		w, err := paramOrErr(doc, "W", "embedding")
		if err != nil {
			return nil, err
		}
		if err := checkParamLen("embedding", "W", len(e.W.Value.Data), w); err != nil {
			return nil, err
		}
		copy(e.W.Value.Data, w.Data)
		return e, nil

	case "conv1d":
		var cfg conv1DConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: conv1d config: %w", err)
		}
		if cfg.InChannels <= 0 || cfg.OutChannels <= 0 || cfg.KernelSize <= 0 {
			return nil, fmt.Errorf("nn: Load: conv1d config: in_channels, out_channels, and kernel_size must be positive, got %d, %d, and %d", cfg.InChannels, cfg.OutChannels, cfg.KernelSize)
		}
		if cfg.Stride == 0 {
			cfg.Stride = 1
		}
		c := newConv1D(rng, cfg.InChannels, cfg.OutChannels, cfg.KernelSize, cfg.Padding, cfg.Stride, ZerosInit())
		w, err := paramOrErr(doc, "W", "conv1d")
		if err != nil {
			return nil, err
		}
		b, err := paramOrErr(doc, "B", "conv1d")
		if err != nil {
			return nil, err
		}
		if err := checkParamLen("conv1d", "W", len(c.W.Value.Data), w); err != nil {
			return nil, err
		}
		if err := checkParamLen("conv1d", "B", len(c.B.Value.Data), b); err != nil {
			return nil, err
		}
		copy(c.W.Value.Data, w.Data)
		copy(c.B.Value.Data, b.Data)
		return c, nil

	case "convtranspose2d":
		var cfg convTranspose2DConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: convtranspose2d config: %w", err)
		}
		if cfg.InChannels <= 0 || cfg.OutChannels <= 0 || cfg.KernelSize <= 0 || cfg.Stride <= 0 {
			return nil, fmt.Errorf("nn: Load: convtranspose2d config: in_channels, out_channels, kernel_size, and stride must be positive, got %d, %d, %d, and %d", cfg.InChannels, cfg.OutChannels, cfg.KernelSize, cfg.Stride)
		}
		c := ConvTranspose2D(rng, cfg.InChannels, cfg.OutChannels, cfg.KernelSize, cfg.Stride, cfg.Padding, ZerosInit())
		w, err := paramOrErr(doc, "W", "convtranspose2d")
		if err != nil {
			return nil, err
		}
		b, err := paramOrErr(doc, "B", "convtranspose2d")
		if err != nil {
			return nil, err
		}
		if err := checkParamLen("convtranspose2d", "W", len(c.W.Value.Data), w); err != nil {
			return nil, err
		}
		if err := checkParamLen("convtranspose2d", "B", len(c.B.Value.Data), b); err != nil {
			return nil, err
		}
		copy(c.W.Value.Data, w.Data)
		copy(c.B.Value.Data, b.Data)
		return c, nil

	case "frozen":
		if len(doc.Modules) != 1 {
			return nil, fmt.Errorf("nn: Load: frozen module must wrap exactly 1 module, got %d", len(doc.Modules))
		}
		inner, err := decodeModule(doc.Modules[0], rng)
		if err != nil {
			return nil, err
		}
		return &FrozenModule{inner: inner}, nil

	case "softmax":
		return Softmax(), nil
	case "relu":
		return ReLU(), nil
	case "sigmoid":
		return Sigmoid(), nil
	case "tanh":
		return Tanh(), nil
	case "gelu":
		return GELU(), nil
	case "leaky_relu":
		var cfg leakyReLUConfig
		if err := json.Unmarshal(doc.Config, &cfg); err != nil {
			return nil, fmt.Errorf("nn: Load: leaky_relu config: %w", err)
		}
		return LeakyReLU(cfg.Alpha), nil

	case "sequential":
		children := make([]Module, len(doc.Modules))
		for i, cd := range doc.Modules {
			cm, err := decodeModule(cd, rng)
			if err != nil {
				return nil, err
			}
			children[i] = cm
		}
		return &SequentialModel{modules: children}, nil

	case "residual":
		children := make([]Module, len(doc.Modules))
		for i, cd := range doc.Modules {
			cm, err := decodeModule(cd, rng)
			if err != nil {
				return nil, err
			}
			children[i] = cm
		}
		var shortcut Module
		if doc.Shortcut != nil {
			sm, err := decodeModule(*doc.Shortcut, rng)
			if err != nil {
				return nil, err
			}
			shortcut = sm
		}
		return &ResidualBlock{inner: children, shortcut: shortcut}, nil

	default:
		return nil, fmt.Errorf("nn: Load: unknown module type %q", doc.Type)
	}
}

// decodeRoot parses JSON and reconstructs the model tree.
func decodeRoot(data []byte, prefix string) (*SequentialModel, error) {
	var doc moduleDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("nn: %s: %w", prefix, err)
	}
	m, err := decodeModule(doc, NewRNG(0))
	if err != nil {
		return nil, err
	}
	seq, ok := m.(*SequentialModel)
	if !ok {
		return nil, fmt.Errorf("nn: %s: root module has type %q, want \"sequential\"", prefix, doc.Type)
	}
	return seq, nil
}

// Load reads a JSON document written by Save and reconstructs the module
// tree with its trained weights. The weight-init RNG passed to
// constructors during reconstruction is never actually used for
// randomness — every Param is immediately overwritten from the saved
// data — so a fixed throwaway seed is fine here.
func Load(path string) (*SequentialModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("nn: Load: %w", err)
	}
	return decodeRoot(data, "Load")
}

// Marshal encodes a model as JSON bytes (non-indented).
// The output can be decoded by Unmarshal.
func Marshal(model *SequentialModel) ([]byte, error) {
	doc, err := encodeRoot(model, "Marshal")
	if err != nil {
		return nil, err
	}
	return json.Marshal(doc)
}

// Unmarshal decodes JSON bytes into a SequentialModel.
// The root module must be of type "sequential".
func Unmarshal(data []byte) (*SequentialModel, error) {
	return decodeRoot(data, "Unmarshal")
}

// NormalizationStats is a per-channel mean/std pair, typically produced
// by data.NormalizeImagesWithStats, bundled into Metadata so a saved
// model carries the exact preprocessing it was trained with.
type NormalizationStats struct {
	Mean []float32 `json:"mean"`
	Std  []float32 `json:"std"`
}

// Metadata is everything about a trained model that isn't a weight or an
// architecture choice, but that inference still needs: what shape to feed
// it, what its output classes mean, and how to preprocess raw input the
// same way training data was preprocessed. All fields are optional.
type Metadata struct {
	InputShape    []int               `json:"input_shape,omitempty"`
	ClassNames    []string            `json:"class_names,omitempty"`
	Normalization *NormalizationStats `json:"normalization,omitempty"`
}

type modelDoc struct {
	Model    moduleDoc `json:"model"`
	Metadata Metadata  `json:"metadata"`
}

// SaveWithMetadata is Save plus an arbitrary Metadata envelope — a
// distinct, opt-in file format from plain Save/Load (the two are not
// cross-compatible: a file written by Save must be read by Load, and one
// written by SaveWithMetadata must be read by LoadWithMetadata). Use it
// when you want a single file that's sufficient to run inference on raw
// input without separately tracking normalization stats or class names.
func SaveWithMetadata(model *SequentialModel, path string, meta Metadata) error {
	doc, err := encodeRoot(model, "SaveWithMetadata")
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(modelDoc{Model: doc, Metadata: meta}, "", "  ")
	if err != nil {
		return fmt.Errorf("nn: SaveWithMetadata: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("nn: SaveWithMetadata: %w", err)
	}
	return nil
}

// LoadWithMetadata reads a file written by SaveWithMetadata.
func LoadWithMetadata(path string) (*SequentialModel, Metadata, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, Metadata{}, fmt.Errorf("nn: LoadWithMetadata: %w", err)
	}
	var full modelDoc
	if err := json.Unmarshal(data, &full); err != nil {
		return nil, Metadata{}, fmt.Errorf("nn: LoadWithMetadata: %w", err)
	}
	m, err := decodeModule(full.Model, NewRNG(0))
	if err != nil {
		return nil, Metadata{}, err
	}
	seq, ok := m.(*SequentialModel)
	if !ok {
		return nil, Metadata{}, fmt.Errorf("nn: LoadWithMetadata: root module has type %q, want \"sequential\"", full.Model.Type)
	}
	return seq, full.Metadata, nil
}

// Clone creates a fully independent deep copy of a model by marshaling
// it to JSON and unmarshaling it back.
func Clone(model *SequentialModel) (*SequentialModel, error) {
	if model == nil {
		return nil, fmt.Errorf("nn: Clone: model is nil")
	}
	data, err := Marshal(model)
	if err != nil {
		return nil, err
	}
	return Unmarshal(data)
}
