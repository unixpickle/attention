package attention

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

type ibInitState struct {
	Inner rnn.State
}

type ibState struct {
	Inner    rnn.State
	Resource autofunc.RFunc
	Query    linalg.Vector
}

type ibInitRState struct {
	Inner rnn.RState
}

type ibRState struct {
	Inner    rnn.RState
	Resource autofunc.RFunc
	Query    linalg.Vector
	RQuery   linalg.Vector
}

type ibStateGrad struct {
	Inner         rnn.StateGrad
	UpstreamQuery linalg.Vector
}

type ibRStateGrad struct {
	Inner          rnn.RStateGrad
	UpstreamQuery  linalg.Vector
	RUpstreamQuery linalg.Vector
}

// An interfacerBlock gives a block acccess to an external
// resource, represented as an autofunc.RFunc.
type interfacerBlock struct {
	// Resources stores one resource per lane, under the
	// assumption that the first timestep of the block will
	// receive len(Resources) inputs.
	Resources []autofunc.RFunc

	// Block is the wrapped block.
	Block rnn.Block

	// ResInSize is the size of the resources' inputs.
	ResInSize int
}

// StartState returns the start state.
func (i *interfacerBlock) StartState() rnn.State {
	return ibInitState{Inner: i.Block.StartState()}
}

// StartRState returns the start state.
func (i *interfacerBlock) StartRState(rv autofunc.RVector) rnn.RState {
	return ibInitRState{Inner: i.Block.StartRState(rv)}
}

// PropagateStart propagates through the inner block's
// start state.
func (i *interfacerBlock) PropagateStart(s []rnn.State, u []rnn.StateGrad, g autofunc.Gradient) {
	innerStates := make([]rnn.State, len(s))
	innerUpstream := make([]rnn.StateGrad, len(s))
	for j, x := range s {
		innerStates[j] = x.(ibInitState).Inner
		innerUpstream[j] = u[j].(ibStateGrad).Inner
	}
	i.Block.PropagateStart(innerStates, innerUpstream, g)
}

// PropagateStartR propagates through the inner block's
// start state.
func (i *interfacerBlock) PropagateStartR(s []rnn.RState, u []rnn.RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	innerStates := make([]rnn.RState, len(s))
	innerUpstream := make([]rnn.RStateGrad, len(s))
	for j, x := range s {
		innerStates[j] = x.(ibInitRState).Inner
		innerUpstream[j] = u[j].(ibRStateGrad).Inner
	}
	i.Block.PropagateStartR(innerStates, innerUpstream, rg, g)
}

// ApplyBlock applies the block to a batch of inputs.
//
// If this is the first timestep (as shown by the states
// in s), then len(s) must equal len(i.Resources).
func (i *interfacerBlock) ApplyBlock(s []rnn.State, in []autofunc.Result) rnn.BlockResult {
	if _, ok := s[0].(ibInitState); ok {
		if len(s) != len(i.Resources) {
			panic("first batch size must match resource count")
		}
		oldS := s
		s = make([]rnn.State, len(s))
		for j := range s {
			s[j] = ibState{
				Inner:    oldS[j].(ibInitState).Inner,
				Resource: i.Resources[j],
				Query:    make(linalg.Vector, i.ResInSize),
			}
		}
	}
	n := len(s)
	pool := make([]*autofunc.Variable, n)
	joinedIns := make([]autofunc.Result, n)
	innerStates := make([]rnn.State, n)
	for j, x := range s {
		state := x.(ibState)
		pool[j] = &autofunc.Variable{Vector: state.Query}
		joinedIns[j] = autofunc.Concat(state.Resource.Apply(pool[j]), in[j])
		innerStates[j] = state.Inner
	}
	blockRes := i.Block.ApplyBlock(innerStates, joinedIns)

	outStates := make([]rnn.State, n)
	outVecs := make([]linalg.Vector, n)
	for j, v := range blockRes.Outputs() {
		outVecs[j] = v[i.ResInSize:]
		outStates[j] = ibState{
			Inner:    blockRes.States()[j],
			Resource: s[j].(ibState).Resource,
			Query:    v[:i.ResInSize],
		}
	}

	return &ibResult{
		ReqSize:   i.ResInSize,
		Result:    blockRes,
		QueryPool: pool,
		OutStates: outStates,
		OutVecs:   outVecs,
	}
}

// ApplyBlock applies the block to a batch of inputs.
//
// If this is the first timestep (as shown by the states
// in s), then len(s) must equal len(i.Resources).
func (i *interfacerBlock) ApplyBlockR(rv autofunc.RVector, s []rnn.RState,
	in []autofunc.RResult) rnn.BlockRResult {
	if _, ok := s[0].(ibInitRState); ok {
		if len(s) != len(i.Resources) {
			panic("first batch size must match resource count")
		}
		oldS := s
		s = make([]rnn.RState, len(s))
		for j := range s {
			s[j] = ibRState{
				Inner:    oldS[j].(ibInitRState).Inner,
				Resource: i.Resources[j],
				Query:    make(linalg.Vector, i.ResInSize),
				RQuery:   make(linalg.Vector, i.ResInSize),
			}
		}
	}
	n := len(s)
	pool := make([]*autofunc.Variable, n)
	joinedIns := make([]autofunc.RResult, n)
	innerStates := make([]rnn.RState, n)
	for j, x := range s {
		state := x.(ibRState)
		pool[j] = &autofunc.Variable{Vector: state.Query}
		rVar := &autofunc.RVariable{
			Variable:   pool[j],
			ROutputVec: state.RQuery,
		}
		joinedIns[j] = autofunc.ConcatR(state.Resource.ApplyR(rv, rVar), in[j])
		innerStates[j] = state.Inner
	}
	blockRes := i.Block.ApplyBlockR(rv, innerStates, joinedIns)

	outStates := make([]rnn.RState, n)
	outVecs := make([]linalg.Vector, n)
	outVecsR := make([]linalg.Vector, n)
	rOuts := blockRes.ROutputs()
	for j, v := range blockRes.Outputs() {
		rv := rOuts[j]
		outVecs[j] = v[i.ResInSize:]
		outVecsR[j] = rv[i.ResInSize:]
		outStates[j] = ibRState{
			Inner:    blockRes.RStates()[j],
			Resource: s[j].(ibRState).Resource,
			Query:    v[:i.ResInSize],
			RQuery:   rv[:i.ResInSize],
		}
	}

	return &ibRResult{
		ReqSize:   i.ResInSize,
		Result:    blockRes,
		QueryPool: pool,
		OutStates: outStates,
		OutVecs:   outVecs,
		ROutVecs:  outVecsR,
	}
}

type ibResult struct {
	ReqSize int

	Result    rnn.BlockResult
	QueryPool []*autofunc.Variable

	OutStates []rnn.State
	OutVecs   []linalg.Vector
}

func (i *ibResult) Outputs() []linalg.Vector {
	return i.OutVecs
}

func (i *ibResult) States() []rnn.State {
	return i.OutStates
}

func (i *ibResult) PropagateGradient(u []linalg.Vector, s []rnn.StateGrad,
	g autofunc.Gradient) []rnn.StateGrad {
	n := len(i.OutVecs)
	if s == nil {
		s = make([]rnn.StateGrad, n)
	}
	innerUp := make([]linalg.Vector, n)
	innerStateUp := make([]rnn.StateGrad, n)
	for j, x := range s {
		if x == nil {
			innerUp[j] = make(linalg.Vector, i.ReqSize)
		} else {
			y := x.(ibStateGrad)
			innerStateUp[j] = y.Inner
			innerUp[j] = y.UpstreamQuery
		}
	}
	if u != nil {
		for j, x := range u {
			innerUp[j] = append(innerUp[j], x...)
		}
	} else {
		for j, x := range i.OutVecs {
			innerUp[j] = append(innerUp[j], make(linalg.Vector, len(x))...)
		}
	}
	for _, v := range i.QueryPool {
		g[v] = make(linalg.Vector, len(v.Vector))
	}
	downStates := i.Result.PropagateGradient(innerUp, innerStateUp, g)
	res := make([]rnn.StateGrad, len(downStates))
	for i, v := range i.QueryPool {
		res[i] = ibStateGrad{
			Inner:         downStates[i],
			UpstreamQuery: g[v],
		}
		delete(g, v)
	}
	return res
}

type ibRResult struct {
	ReqSize int

	Result    rnn.BlockRResult
	QueryPool []*autofunc.Variable

	OutStates []rnn.RState
	OutVecs   []linalg.Vector
	ROutVecs  []linalg.Vector
}

func (i *ibRResult) Outputs() []linalg.Vector {
	return i.OutVecs
}

func (i *ibRResult) ROutputs() []linalg.Vector {
	return i.ROutVecs
}

func (i *ibRResult) RStates() []rnn.RState {
	return i.OutStates
}

func (i *ibRResult) PropagateRGradient(u, uR []linalg.Vector, s []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []rnn.RStateGrad {
	n := len(i.OutVecs)
	if s == nil {
		s = make([]rnn.RStateGrad, n)
	}
	if g == nil {
		g = autofunc.Gradient{}
	}
	innerUp := make([]linalg.Vector, n)
	innerUpR := make([]linalg.Vector, n)
	innerStateUp := make([]rnn.RStateGrad, n)
	for j, x := range s {
		if x == nil {
			innerUp[j] = make(linalg.Vector, i.ReqSize)
			innerUpR[j] = make(linalg.Vector, i.ReqSize)
		} else {
			y := x.(ibRStateGrad)
			innerStateUp[j] = y.Inner
			innerUp[j] = y.UpstreamQuery
			innerUpR[j] = y.RUpstreamQuery
		}
	}
	if u != nil {
		for j, x := range u {
			innerUp[j] = append(innerUp[j], x...)
			innerUpR[j] = append(innerUpR[j], uR[j]...)
		}
	} else {
		for j, x := range i.OutVecs {
			zeroVec := make(linalg.Vector, len(x))
			innerUp[j] = append(innerUp[j], zeroVec...)
			innerUpR[j] = append(innerUpR[j], zeroVec...)
		}
	}
	for _, v := range i.QueryPool {
		g[v] = make(linalg.Vector, len(v.Vector))
		rg[v] = make(linalg.Vector, len(v.Vector))
	}
	downStates := i.Result.PropagateRGradient(innerUp, innerUpR, innerStateUp, rg, g)
	res := make([]rnn.RStateGrad, len(downStates))
	for i, v := range i.QueryPool {
		res[i] = ibRStateGrad{
			Inner:          downStates[i],
			UpstreamQuery:  g[v],
			RUpstreamQuery: rg[v],
		}
		delete(g, v)
		delete(rg, v)
	}
	return res
}
