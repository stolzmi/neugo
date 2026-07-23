// nn/convention_test.go
package nn

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// recvTypeName extracts the bare type name a method is declared on (e.g.
// "LinearLayer" from "func (l *LinearLayer) OutputShape(...)").
func recvTypeName(recv *ast.FieldList) string {
	if recv == nil || len(recv.List) == 0 {
		return ""
	}
	expr := recv.List[0].Type
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	if ident, ok := expr.(*ast.Ident); ok {
		return ident.Name
	}
	return ""
}

// starPointerTypeName returns "TypeName" for a function's results list
// holding a single "*TypeName" return value, or "" for anything else
// (named return, multiple returns, non-pointer, etc.).
func starPointerTypeName(results *ast.FieldList) string {
	if results == nil || len(results.List) != 1 || len(results.List[0].Names) > 1 {
		return ""
	}
	star, ok := results.List[0].Type.(*ast.StarExpr)
	if !ok {
		return ""
	}
	ident, ok := star.X.(*ast.Ident)
	if !ok {
		return ""
	}
	return ident.Name
}

// moduleInventory is everything findModuleTypes learns from one parse
// pass: the set of types that look like Module implementations (declare
// their own OutputShape — this can't see a method promoted from an
// embedded field, e.g. InstanceNormLayer's OutputShape comes from its
// embedded *GroupNormLayer, so InstanceNormLayer itself is invisible to
// this scan; a known, acceptable gap — it makes this check under-count,
// never over-count), plus, for each such type, every top-level
// constructor function that returns exactly a "*TypeName" — since callers
// almost always spell the constructor name (RNN(...), Conv2D(...)), not
// the underlying type name (*RNNLayer, *Conv2DLayer), test-coverage
// detection below checks for the constructor names, not the type name.
type moduleInventory struct {
	types        map[string]bool
	constructors map[string][]string // type name -> constructor function names returning *that type
}

func findModuleTypes(t *testing.T, dir string) moduleInventory {
	t.Helper()
	fset := token.NewFileSet()
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}

	inv := moduleInventory{types: map[string]bool{}, constructors: map[string][]string{}}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		node, err := parser.ParseFile(fset, f, nil, 0)
		if err != nil {
			t.Fatalf("parsing %s: %v", f, err)
		}
		for _, decl := range node.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			if fn.Recv != nil {
				if fn.Name.Name == "OutputShape" {
					if name := recvTypeName(fn.Recv); name != "" {
						inv.types[name] = true
					}
				}
				continue
			}
			// A top-level function: candidate constructor if it returns
			// exactly "*SomeType" and its name starts with an uppercase
			// letter (exported).
			if !fn.Name.IsExported() {
				continue
			}
			if typeName := starPointerTypeName(fn.Type.Results); typeName != "" {
				inv.constructors[typeName] = append(inv.constructors[typeName], fn.Name.Name)
			}
		}
	}
	return inv
}

// TestEveryModuleTypeHasSerializationAndTests is a convention-enforcement
// check, not a correctness one: every layer type this package adds that
// implements Module (detected via its own OutputShape declaration) must
// have a matching case in serialize.go's encodeModule switch, and its
// constructor(s) must be referenced by at least one _test.go file — a
// cheap proxy for "has a gradcheck/serialize-roundtrip test", not a
// guarantee one exists. The goal is to catch a forgotten serialize.go
// case or test file the moment a new layer is added, the same way the
// existing per-layer tests already enforce gradient correctness.
func TestEveryModuleTypeHasSerializationAndTests(t *testing.T) {
	inv := findModuleTypes(t, ".")

	serializeSrc, err := os.ReadFile("serialize.go")
	if err != nil {
		t.Fatal(err)
	}
	serializeSrcStr := string(serializeSrc)

	var testSrc strings.Builder
	testFiles, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range testFiles {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		testSrc.Write(b)
	}
	testSrcStr := testSrc.String()

	// SequentialModel is the composite root wrapper, not a leaf layer —
	// it's handled specially in serialize.go and is exercised by every
	// single serialization test in the package, so it's exempt from
	// this per-leaf-layer check rather than a genuine gap.
	delete(inv.types, "SequentialModel")

	for typeName := range inv.types {
		if !strings.Contains(serializeSrcStr, "*"+typeName+":") {
			t.Errorf("type %s implements Module (has its own OutputShape) but has no case in serialize.go's encodeModule switch", typeName)
		}

		constructors := inv.constructors[typeName]
		covered := strings.Contains(testSrcStr, typeName) // catches direct type-assertion usage too
		for _, ctor := range constructors {
			if strings.Contains(testSrcStr, ctor+"(") {
				covered = true
				break
			}
		}
		if !covered {
			t.Errorf("type %s implements Module but neither it nor any of its constructors %v are referenced in any _test.go file", typeName, constructors)
		}
	}
}
