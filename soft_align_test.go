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

type softAlignTestFunc struct {
	SA *SoftAlign
}

func (s softAlignTestFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	return s.SA.Apply(in, seqfunc.ConstResult(s.decInVecs()))
}

func (s softAlignTestFunc) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	return s.SA.ApplyR(rv, in, seqfunc.ConstRResult(s.decInVecs()))
}

func (s softAlignTestFunc) decInVecs() [][]linalg.Vector {
	return [][]linalg.Vector{
		{{1}, {-1}, {0}, {0}, {0.5}},
		{{-1}, {-0.7}, {0.5}},
		{{0.3}, {0.2}},
	}
}

func TestSoftAlign(t *testing.T) {
	decoder := rnn.NewLSTM(3, 4)
	encoder := rnn.NewLSTM(1, 2)
	attentor := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  5,
			OutputCount: 1,
		},
	}
	attentor.Randomize()
	startQuery := &autofunc.Variable{Vector: []float64{0, -1, 1}}
	params := append([]*autofunc.Variable{startQuery}, decoder.Parameters()...)
	params = append(params, encoder.Parameters()...)
	params = append(params, attentor.Parameters()...)
	inputs := [][]*autofunc.Variable{
		{{Vector: []float64{1}}, {Vector: []float64{-1}}, {Vector: []float64{1}}},
		{{Vector: []float64{0.5}}, {Vector: []float64{-0.5}}},
		{{Vector: []float64{0.5}}, {Vector: []float64{-1}}, {Vector: []float64{0.5}}},
	}
	params = append(params, startQuery)
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
		F: &softAlignTestFunc{
			SA: &SoftAlign{
				Encoder:    &rnn.BlockSeqFunc{B: encoder},
				Decoder:    decoder,
				Attentor:   attentor,
				BatchSize:  2,
				StartQuery: startQuery,
			},
		},
		Vars:  params,
		Input: inputs,
		RV:    rv,
	}
	checker.FullCheck(t)
}
