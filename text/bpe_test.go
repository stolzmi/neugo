package text

import (
	"path/filepath"
	"testing"
)

func TestTrainBPERoundTripEncodeDecode(t *testing.T) {
	corpus := []string{
		"the quick brown fox jumps over the lazy dog",
		"the quick brown fox is quick",
	}
	tok := TrainBPE(corpus, 300)

	for _, s := range []string{
		"the quick brown fox",
		"a completely different sentence, with punctuation!",
		"   leading and trailing whitespace   ",
	} {
		ids := tok.Encode(s)
		got := tok.Decode(ids)
		if got != s {
			t.Errorf("Decode(Encode(%q)) = %q, want %q", s, got, s)
		}
	}
}

func TestTrainBPERoundTripUnseenUnicodeText(t *testing.T) {
	// Byte-level BPE has no "unknown token": text using bytes/runes never
	// seen during training must still round-trip exactly.
	tok := TrainBPE([]string{"hello world"}, 300)
	s := "héllo wörld 世界 🎉"
	got := tok.Decode(tok.Encode(s))
	if got != s {
		t.Errorf("Decode(Encode(%q)) = %q, want %q", s, got, s)
	}
}

func TestTrainBPELearnsMerges(t *testing.T) {
	// "ab" repeats far more than any other pair, so it should be merged
	// into a new token, and VocabSize should grow past the base 256.
	var corpus []string
	for i := 0; i < 50; i++ {
		corpus = append(corpus, "ababababab")
	}
	tok := TrainBPE(corpus, 300)
	if tok.VocabSize() <= 256 {
		t.Fatalf("VocabSize() = %d, want > 256 (expected at least one merge to be learned)", tok.VocabSize())
	}

	// A merged "ab" should encode to noticeably fewer tokens than the raw
	// byte count (10 bytes for "ababababab").
	ids := tok.Encode("ababababab")
	if len(ids) >= 10 {
		t.Errorf("len(Encode(\"ababababab\")) = %d, want < 10 (merges should have compressed it)", len(ids))
	}
}

func TestVocabSizeRespectsMinimumOf256(t *testing.T) {
	tok := TrainBPE([]string{"a"}, 10) // vocabSize smaller than 256
	if tok.VocabSize() < 256 {
		t.Errorf("VocabSize() = %d, want >= 256 (base byte vocabulary)", tok.VocabSize())
	}
}

func TestEncodeEmptyStringReturnsEmpty(t *testing.T) {
	tok := TrainBPE([]string{"hello"}, 300)
	ids := tok.Encode("")
	if len(ids) != 0 {
		t.Errorf("Encode(\"\") returned %d ids, want 0", len(ids))
	}
	if got := tok.Decode(ids); got != "" {
		t.Errorf("Decode(Encode(\"\")) = %q, want \"\"", got)
	}
}

func TestSaveLoadBPERoundTrip(t *testing.T) {
	corpus := []string{"the quick brown fox jumps over the lazy dog", "the quick fox"}
	tok := TrainBPE(corpus, 300)

	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := tok.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadBPE(path)
	if err != nil {
		t.Fatalf("LoadBPE: %v", err)
	}

	if loaded.VocabSize() != tok.VocabSize() {
		t.Errorf("loaded VocabSize() = %d, want %d", loaded.VocabSize(), tok.VocabSize())
	}

	s := "the quick brown fox"
	want := tok.Encode(s)
	got := loaded.Encode(s)
	if len(want) != len(got) {
		t.Fatalf("loaded Encode(%q) = %v (len %d), want %v (len %d)", s, got, len(got), want, len(want))
	}
	for i := range want {
		if want[i] != got[i] {
			t.Errorf("token[%d] = %d, want %d", i, got[i], want[i])
		}
	}
	if loaded.Decode(got) != s {
		t.Errorf("loaded round trip = %q, want %q", loaded.Decode(got), s)
	}
}

func TestLoadBPEMissingFileReturnsError(t *testing.T) {
	if _, err := LoadBPE(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("expected error loading a nonexistent file, got nil")
	}
}
