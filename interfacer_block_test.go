package attention

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func TestInterfacerBlock(t *testing.T) {
	resource := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  3,
			OutputCount: 1,
		},
	}
	resource.Randomize()
	block := rnn.NewLSTM(2, 4)
	sf := &rnn.BlockSeqFunc{
		B: &interfacerBlock{
			Resources: []autofunc.RFunc{resource, resource, resource},
			Block:     block,
			ResInSize: 3,
		},
	}
	params := append([]*autofunc.Variable{}, resource.Parameters()...)
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
