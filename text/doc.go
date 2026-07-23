/*
Package text provides dependency-free tokenization utilities for pairing
with nn.Embedding/nn.TransformerBlock-style models: a byte-level
Byte-Pair-Encoding (BPE) tokenizer (GPT-2 style) and a plain line-delimited
corpus loader.

Train a tokenizer from a corpus and use it:

	tok := text.TrainBPE(corpus, 1000) // vocab size 1000
	ids := tok.Encode("hello, world")
	s := tok.Decode(ids) // == "hello, world"

Save and reload a trained tokenizer:

	err := tok.Save("tokenizer.json")
	tok2, err := text.LoadBPE("tokenizer.json")

Load a line-delimited text corpus:

	lines, err := text.LoadLineDataset("corpus.txt")
*/
package text
