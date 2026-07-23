// text/bpe.go
package text

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"unicode"
)

// splitSegments splits s into a sequence of maximal runs where each run is
// entirely whitespace or entirely non-whitespace, in order — e.g.
// "hello, world" -> ["hello,", " ", "world"]. BPE merges never cross a
// whitespace/non-whitespace boundary (a much simpler version of GPT-2's
// pre-tokenization regex); reassembling by plain concatenation, byte for
// byte, in the same order, is exactly why Decode(Encode(s)) == s always
// holds regardless of spacing — no whitespace information is ever
// discarded, just prevented from being merged into a token alongside
// non-whitespace bytes.
func splitSegments(s string) []string {
	if s == "" {
		return nil
	}
	runes := []rune(s)
	var segments []string
	i := 0
	for i < len(runes) {
		j := i
		isSpace := unicode.IsSpace(runes[i])
		for j < len(runes) && unicode.IsSpace(runes[j]) == isSpace {
			j++
		}
		segments = append(segments, string(runes[i:j]))
		i = j
	}
	return segments
}

func isWhitespaceSegment(seg string) bool {
	for _, r := range seg {
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return true
}

// mergeRule records one learned BPE merge: the pair (a, b) merges into
// NewID, and Rank is that merge's position in training order (lower Rank
// = learned earlier = applied first when encoding, since an earlier merge
// can create a token a later merge then combines further).
type mergeRule struct {
	Rank  int
	NewID int
}

// BPETokenizer is a byte-level BPE tokenizer (Sennrich et al., 2016,
// applied at the byte level as in GPT-2): the base vocabulary is exactly
// the 256 possible byte values (ids 0-255), and every learned merge
// combines two existing token ids into one new one (ids 256, 257, ...).
// Operating on raw bytes rather than Unicode codepoints means every
// possible input string is representable — there's no "unknown token"
// case to handle.
type BPETokenizer struct {
	merges    map[[2]int]mergeRule
	mergeList [][2]int // same merges, in learned order — the serialized form
	vocab     map[int][]byte
	nextID    int
}

// mergePairInSeq replaces every non-overlapping adjacent occurrence of
// pair in seq with newID, scanning left to right.
func mergePairInSeq(seq []int, pair [2]int, newID int) []int {
	out := make([]int, 0, len(seq))
	i := 0
	for i < len(seq) {
		if i+1 < len(seq) && seq[i] == pair[0] && seq[i+1] == pair[1] {
			out = append(out, newID)
			i += 2
		} else {
			out = append(out, seq[i])
			i++
		}
	}
	return out
}

func buildMergeMap(merges [][2]int) map[[2]int]mergeRule {
	m := make(map[[2]int]mergeRule, len(merges))
	for i, pair := range merges {
		m[pair] = mergeRule{Rank: i, NewID: 256 + i}
	}
	return m
}

// TrainBPE learns a byte-level BPE vocabulary of size vocabSize (clamped
// up to at least 256, the base byte vocabulary) from corpus: it counts
// whitespace-delimited word frequencies (via splitSegments, skipping
// whitespace-only segments), then repeatedly merges the most frequent
// adjacent token pair across all words — weighted by word frequency —
// into a new token, stopping early if no pair occurs more than once or
// vocabSize is reached.
func TrainBPE(corpus []string, vocabSize int) *BPETokenizer {
	if vocabSize < 256 {
		vocabSize = 256
	}

	wordFreq := map[string]int{}
	for _, doc := range corpus {
		for _, seg := range splitSegments(doc) {
			if isWhitespaceSegment(seg) {
				continue
			}
			wordFreq[seg]++
		}
	}

	words := make([][]int, 0, len(wordFreq))
	freqs := make([]int, 0, len(wordFreq))
	for w, f := range wordFreq {
		bs := []byte(w)
		seq := make([]int, len(bs))
		for i, by := range bs {
			seq[i] = int(by)
		}
		words = append(words, seq)
		freqs = append(freqs, f)
	}

	vocab := make(map[int][]byte, 256)
	for i := 0; i < 256; i++ {
		vocab[i] = []byte{byte(i)}
	}

	type pairCount struct {
		pair  [2]int
		count int
	}

	var merges [][2]int
	nextID := 256

	for nextID < vocabSize {
		counts := map[[2]int]int{}
		for wi, seq := range words {
			f := freqs[wi]
			for i := 0; i+1 < len(seq); i++ {
				counts[[2]int{seq[i], seq[i+1]}] += f
			}
		}
		if len(counts) == 0 {
			break
		}

		pcs := make([]pairCount, 0, len(counts))
		for p, c := range counts {
			pcs = append(pcs, pairCount{p, c})
		}
		// Deterministic regardless of map iteration order: highest count
		// first, ties broken by pair value.
		sort.Slice(pcs, func(i, j int) bool {
			if pcs[i].count != pcs[j].count {
				return pcs[i].count > pcs[j].count
			}
			if pcs[i].pair[0] != pcs[j].pair[0] {
				return pcs[i].pair[0] < pcs[j].pair[0]
			}
			return pcs[i].pair[1] < pcs[j].pair[1]
		})
		if pcs[0].count < 2 {
			break // merging a pair that occurs once buys nothing
		}
		bestPair := pcs[0].pair

		vocab[nextID] = append(append([]byte{}, vocab[bestPair[0]]...), vocab[bestPair[1]]...)
		merges = append(merges, bestPair)
		for wi, seq := range words {
			words[wi] = mergePairInSeq(seq, bestPair, nextID)
		}
		nextID++
	}

	return &BPETokenizer{merges: buildMergeMap(merges), mergeList: merges, vocab: vocab, nextID: nextID}
}

// applyMerges repeatedly finds the lowest-rank (earliest-learned) merge
// whose pair currently appears adjacent somewhere in tokens and applies
// it — the standard BPE encode loop — until no learned merge applies.
func (b *BPETokenizer) applyMerges(tokens []int) []int {
	for {
		bestRank := -1
		bestIdx := -1
		for i := 0; i+1 < len(tokens); i++ {
			pair := [2]int{tokens[i], tokens[i+1]}
			if rule, ok := b.merges[pair]; ok {
				if bestRank == -1 || rule.Rank < bestRank {
					bestRank = rule.Rank
					bestIdx = i
				}
			}
		}
		if bestIdx == -1 {
			break
		}
		pair := [2]int{tokens[bestIdx], tokens[bestIdx+1]}
		tokens = mergePairInSeq(tokens, pair, b.merges[pair].NewID)
	}
	return tokens
}

// Encode tokenizes s: splits into whitespace/non-whitespace segments
// (see splitSegments), converts each segment to its raw bytes, and
// applies the learned merges within each segment independently.
func (b *BPETokenizer) Encode(s string) []int {
	var ids []int
	for _, seg := range splitSegments(s) {
		bs := []byte(seg)
		seq := make([]int, len(bs))
		for i, by := range bs {
			seq[i] = int(by)
		}
		ids = append(ids, b.applyMerges(seq)...)
	}
	return ids
}

// Decode reverses Encode: every token id maps back to its byte sequence
// (from the base 256 bytes or a learned merge's concatenation), and those
// are concatenated in order — always reproducing the original bytes
// exactly, since Encode never discards any input byte.
func (b *BPETokenizer) Decode(ids []int) string {
	var buf []byte
	for _, id := range ids {
		if bs, ok := b.vocab[id]; ok {
			buf = append(buf, bs...)
		}
	}
	return string(buf)
}

// VocabSize returns the total number of token ids (256 base bytes plus
// every learned merge).
func (b *BPETokenizer) VocabSize() int { return b.nextID }

type bpeDoc struct {
	Merges [][2]int `json:"merges"`
}

// Save writes the tokenizer's learned merges (in order) as JSON — the
// base 256-byte vocabulary is never serialized since it's always
// identical and rebuilt on Load, matching nn.Save/nn.Load's "encode only
// what varies" convention.
func (b *BPETokenizer) Save(path string) error {
	data, err := json.MarshalIndent(bpeDoc{Merges: b.mergeList}, "", "  ")
	if err != nil {
		return fmt.Errorf("text: Save: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("text: Save: %w", err)
	}
	return nil
}

// LoadBPE reads a tokenizer written by Save, rebuilding its vocabulary
// from the base 256 bytes plus the saved merges in order.
func LoadBPE(path string) (*BPETokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("text: LoadBPE: %w", err)
	}
	var doc bpeDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("text: LoadBPE: %w", err)
	}

	vocab := make(map[int][]byte, 256+len(doc.Merges))
	for i := 0; i < 256; i++ {
		vocab[i] = []byte{byte(i)}
	}
	nextID := 256
	for _, pair := range doc.Merges {
		p0, ok0 := vocab[pair[0]]
		p1, ok1 := vocab[pair[1]]
		if !ok0 || !ok1 {
			return nil, fmt.Errorf("text: LoadBPE: merge %v references a token id not yet defined", pair)
		}
		vocab[nextID] = append(append([]byte{}, p0...), p1...)
		nextID++
	}

	return &BPETokenizer{merges: buildMergeMap(doc.Merges), mergeList: doc.Merges, vocab: vocab, nextID: nextID}, nil
}
