package attention

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func TestInterfacerBlockOutputs(t *testing.T) {
	queries := make([]autofunc.RFunc, 3)
	for i := range queries {
		query := neuralnet.Network{
			&neuralnet.DenseLayer{
				InputCount:  3,
				OutputCount: 4,
			},
		}
		query.Randomize()
		queries[i] = query
	}

	initQuery := &autofunc.Variable{Vector: []float64{1, -0.5, 1}}

	block := rnn.NewLSTM(6, 4)

	inSeqs := [][]linalg.Vector{
		{{1, 2}, {3, 0.5}, {-4, 2}},
		{{-1, 1}},
		{{0, -1}, {-3, 4}},
	}

	expectedOuts := make([][]linalg.Vector, len(inSeqs))
	for lane, in := range inSeqs {
		var outs []linalg.Vector
		query := initQuery.Vector
		state := block.StartState()
		for _, vec := range in {
			queryRes := queries[lane].Apply(&autofunc.Variable{Vector: query}).Output()
			blockIn := append(queryRes, vec...)
			out := block.ApplyBlock([]rnn.State{state}, []autofunc.Result{
				&autofunc.Variable{Vector: blockIn},
			})
			state = out.States()[0]
			query = out.Outputs()[0][:3]
			outs = append(outs, out.Outputs()[0][3:])
		}
		expectedOuts[lane] = outs
	}

	ib := &interfacerBlock{
		Resources:  queries,
		Block:      block,
		StartQuery: initQuery,
	}

	sf := rnn.BlockSeqFunc{B: ib}
	actualOuts := sf.ApplySeqs(seqfunc.ConstResult(inSeqs)).OutputSeqs()

	if !sequencesClose(expectedOuts, actualOuts) {
		t.Errorf("expected %v but got %v", expectedOuts, actualOuts)
	}
}

func TestInterfacerBlock(t *testing.T) {
	resource := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  3,
			OutputCount: 1,
		},
	}
	resource.Randomize()
	block := rnn.NewLSTM(2, 4)
	startQuery := &autofunc.Variable{Vector: []float64{-1, 0, 1}}
	sf := &rnn.BlockSeqFunc{
		B: &interfacerBlock{
			Resources:  []autofunc.RFunc{resource, resource, resource},
			Block:      block,
			StartQuery: startQuery,
		},
	}
	params := append([]*autofunc.Variable{startQuery}, resource.Parameters()...)
	params = append(params, block.Parameters()...)
	inputs := [][]*autofunc.Variable{
		{{Vector: []float64{1}}, {Vector: []float64{-1}}, {Vector: []float64{1}}},
		{{Vector: []float64{0.5}}},
		{{Vector: []float64{0.5}}, {Vector: []float64{1}}},
	}
	for _, v := range inputs {
		for _, u := range v {
			params = append(params, u)
		}
	}
	rv := autofunc.RVector{}
	for _, v := range params {
		rv[v] = make(linalg.Vector, len(v.Vector))
		for i := range rv[v] {
			rv[v][i] = rand.NormFloat64()
		}
	}
	checker := functest.SeqRFuncChecker{
		F:     sf,
		Vars:  params,
		Input: inputs,
		RV:    rv,
	}
	checker.FullCheck(t)
}

func sequencesClose(s1, s2 [][]linalg.Vector) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, x := range s1 {
		y := s2[i]
		if len(x) != len(y) {
			return false
		}
		for j, w := range x {
			u := y[j]
			if len(u) != len(w) {
				return false
			}
			if w.Copy().Scale(-1).Add(u).MaxAbs() > 1e-5 {
				return false
			}
		}
	}
	return true
}
