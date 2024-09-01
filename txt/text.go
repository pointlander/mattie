// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package txt

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"

	"github.com/pointlander/matrix"
	"github.com/pointlander/matrix/vector"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"

	"github.com/alixaxel/pagerank"
)

const (
	// Symbols
	Symbols = ('z' - 'a' + 1) + ('Z' - 'A' + 1) + 3
	// Size is the link size
	Size = 8
	// Input is the network input size
	Input = Size + 2*Size
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// Scale is the scale of the search
	Scale = 33 //96
	// SetSize is the size of a symbol set
	SetSize = 4
	// Samples is the number of samplee
	Samples = Scale * (Scale - 1) / 2
)

var (
	// To converts to a code
	To = make(map[byte]int, Symbols)
	// From converts from a code
	From = make(map[int]byte, Symbols)
)

func init() {
	index := 0
	for i := 'a'; i <= 'z'; i++ {
		To[byte(i)] = index
		From[index] = byte(i)
		index++
	}
	for i := 'A'; i <= 'Z'; i++ {
		To[byte(i)] = index
		From[index] = byte(i)
		index++
	}
	i := '.'
	To[byte(i)] = index
	From[index] = byte(i)
	index++
	i = ','
	To[byte(i)] = index
	From[index] = byte(i)
	index++
	i = ' '
	To[byte(i)] = index
	From[index] = byte(i)
	index++
}

// PageRank computes the page rank of Q, K
func PageRank(Q, K matrix.Matrix) []float32 {
	graph := pagerank.NewGraph()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		aa := 0.0
		for _, v := range K {
			aa += float64(v * v)
		}
		aa = math.Sqrt(aa)
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			bb := 0.0
			for _, v := range Q {
				bb += float64(v * v)
			}
			bb = math.Sqrt(bb)
			d := float64(vector.Dot(K, Q)) / (aa * bb)
			if d < 0 {
				graph.Link(uint32(i), uint32(j), -d)
			} else {
				graph.Link(uint32(j), uint32(i), d)
			}

		}
	}
	ranks := make([]float32, K.Rows)
	graph.Rank(0.85, 1e-6, func(node uint32, rank float64) {
		ranks[node] = float32(rank)
	})
	return ranks
}

// Set is a set of examples
type Set struct {
	Text []byte
}

// Sets is many sets
type Sets []Set

// Load load text data
func Load() Sets {
	sets := make(Sets, 6)
	input, err := os.Open("1513.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer input.Close()
	decoded := bzip2.NewReader(input)
	data, err := io.ReadAll(decoded)
	if err != nil {
		panic(err)
	}
	sets[0].Text = data

	sets[1].Text = []byte("abcdabcda")
	sets[2].Text = []byte("abcdabcdab")
	sets[3].Text = []byte("abcdabcdabc")
	sets[4].Text = []byte("abcdabcdabcd")
	sets[5].Text = []byte("abcddcbaabcddcbaabcddcbaabcd")
	return sets
}

// Problem is an optimization problem
type Problem struct {
	Opt    matrix.Matrix
	Input  []byte
	Output []byte
	Count  int
}

// Size is the size of the input
func (p Problem) Size() int {
	return p.Count
}

// GetSingleTrainingData gets the training data
func (sets Sets) GetSingleTrainingData(s int) Problem {
	if s == 0 {
		set := sets[s]
		txt := make([]byte, len(set.Text[2048:2048+512]))
		copy(txt, set.Text[2048:2048+512])
		problem := Problem{
			Input:  txt,
			Output: txt[511 : 511+1],
			Count:  512 + 1,
		}
		problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
		index := 0
		for i := 0; i < len(txt); i++ {
			problem.Opt.Data[index+To[txt[i]]] = 1
			index += Input
		}
		return problem
	}
	set := sets[s]
	txt := make([]byte, len(set.Text))
	copy(txt, set.Text)
	problem := Problem{
		Input:  txt[:len(txt)-1],
		Output: txt[len(txt)-1:],
		Count:  len(txt),
	}
	problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
	return problem
}

// Model model is the random matrix model
type Model struct {
	Projection matrix.CompressedRandomMatrix
	Order      matrix.CompressedRandomMatrix
	Symbol     matrix.CompressedRandomMatrix
}

// Sample is a sample
type Sample struct {
	Rng    *matrix.Rand
	A      matrix.CompressedGenerator
	B      matrix.CompressedGenerator
	Order  matrix.CompressedGenerator
	Symbol matrix.CompressedGenerator
	S      int
	Ranks  []float32
	Meta   float32
	Avg    float64
	Stddev float64
}

// Search searches for a symbol
func Search(sets Sets, s int, seed uint32) []Sample {
	opt := sets.GetSingleTrainingData(s)
	model := Model{
		Projection: matrix.NewCompressedRandomMatrix(Input, Input),
		Order:      matrix.NewCompressedRandomMatrix(Size, opt.Size()),
		Symbol:     matrix.NewCompressedRandomMatrix(Size, Symbols),
	}
	rng := matrix.Rand(seed)
	projections := make([]matrix.CompressedGenerator, Scale)
	for i := range projections {
		projections[i] = model.Projection.Sample(&rng)
	}
	index := 0
	samples := make([]Sample, Samples)
	for i := 0; i < Scale; i++ {
		for j := i + 1; j < Scale; j++ {
			Rng := matrix.Rand(seed)
			order := model.Order.Sample(&Rng)
			symbol := model.Symbol.Sample(&Rng)
			/*seed := rng.Uint32()
			if seed == 0 {
				seed += 1
			}*/
			samples[index].Rng = &Rng
			samples[index].A = projections[i]
			samples[index].B = projections[j]
			samples[index].Order = order
			samples[index].Symbol = symbol
			samples[index].S = index % SetSize
			index++
		}
	}
	done := make(chan bool, 8)
	process := func(sample *Sample) {
		opt := sets.GetSingleTrainingData(s)
		/*rng := sample.Rng
		order := make([]float32, Size)
		for i := range order {
			order[i] = float32(rng.NormFloat64())
		}
		for i := 0; i < opt.Opt.Rows; i++ {
			copy(opt.Opt.Data[i*Input+Size:i*Input+Size+Size], order)
			for j, value := range order {
				order[j] = (value + float32(rng.NormFloat64())) / 2
			}
		}*/
		order := sample.Order.Sample()
		a, b := 0, 1
		jj := opt.Opt.Rows
		for j := 0; j < jj; j++ {
			x, y := (j+a)%opt.Opt.Rows, (j+b)%opt.Opt.Rows
			copy(opt.Opt.Data[j*Input+Size:j*Input+Size+Size],
				order.Data[x*Size:(x+1)*Size])
			copy(opt.Opt.Data[j*Input+Size+Size:j*Input+Size+2*Size],
				order.Data[(y)*Size:(y+1)*Size])
			a, b = b, a
		}
		/*if x := jj + a; x < opt.Opt.Rows {
			copy(opt.Opt.Data[jj*Input+Size:jj*Input+Size+Size],
				order.Data[x*Size:(x+1)*Size])
		}
		if y := jj + b; y < opt.Opt.Rows {
			copy(opt.Opt.Data[jj*Input+Size+Size:jj*Input+Size+2*Size],
				order.Data[(y)*Size:(y+1)*Size])
		}*/
		syms := sample.Symbol.Sample()
		index := 0
		for i := 0; i < len(opt.Input); i++ {
			symbol := syms.Data[Size*To[opt.Input[i]] : Size*(To[opt.Input[i]]+1)]
			copy(opt.Opt.Data[index:index+Input], symbol)
			index += Input
		}
		params := opt.Opt.Data[Input*(opt.Size()-1) : Input*(opt.Size())]
		symbol := syms.Data[Size*To[byte(sample.S)] : Size*(To[byte(sample.S)]+1)]
		copy(params, symbol)
		/*factor := 1.0 / float64(Input)
		for i := range opt.Opt.Data {
			opt.Opt.Data[i] += float32(factor * sample.Rng.Float64())
		}*/
		query := sample.A.Sparse()
		key := sample.B.Sparse()
		q := query.MulT(opt.Opt)
		k := key.MulT(opt.Opt)
		sample.Ranks = PageRank(q, k)
		done <- true
	}
	flight, index, cpus := 0, 0, runtime.NumCPU()
	for flight < cpus && index < len(samples) {
		sample := &samples[index]
		go process(sample)
		index++
		flight++
	}
	for index < len(samples) {
		<-done
		flight--

		sample := &samples[index]
		go process(sample)
		index++
		flight++
	}
	for i := 0; i < flight; i++ {
		<-done
	}

	return samples
}

// Text mode
func Text(full bool, s int, seed uint32) int {
	sets := Load()
	opt := sets.GetSingleTrainingData(s)
	fmt.Println(string(opt.Input))
	fmt.Println(string(opt.Output))
	samples := Search(sets, s, seed)
	r := matrix.NewMatrix(len(samples[0].Ranks), len(samples))
	input := make([]float64, 0, len(samples[0].Ranks)*len(samples))
	for sample := range samples {
		ranks := samples[sample].Ranks
		for _, rank := range ranks {
			r.Data = append(r.Data, float32(rank))
			input = append(input, float64(rank))
		}
	}
	x := mat.NewDense(len(samples), len(samples[0].Ranks), input)
	dst := mat.SymDense{}
	stat.CovarianceMatrix(&dst, x, nil)
	rr, cc := dst.Dims()
	fmt.Println(rr, cc)
	fa := mat.Formatted(&dst, mat.Squeeze())
	fmt.Println(fa)
	graph := pagerank.NewGraph()
	for i := 0; i < rr; i++ {
		for j := 0; j < cc; j++ {
			d := dst.At(i, j)
			if d > 0 {
				graph.Link(uint32(i), uint32(j), d)
				graph.Link(uint32(j), uint32(i), d)
			}
		}
	}
	ranks := make([]float64, rr)
	graph.Rank(1, 1e-6, func(node uint32, rank float64) {
		ranks[node] = float64(rank)
	})
	fmt.Println("ranks", ranks)
	results := make([]float64, 4)
	counts := make([]float64, 4)
	for i, v := range opt.Input {
		results[To[v]] += ranks[i+1]
		counts[To[v]]++
	}
	for i := range results {
		results[i] /= counts[i]
	}
	fmt.Println("results", results)
	/*symbol, m := 0, 0.0
	for i := 1; i < cc-1; i++ {
		value := dst.At(rr-1, i)
		if value > m {
			m, symbol = value, i
			fmt.Printf("symbol %c %f\n", opt.Input[symbol-1], m)
		}
	}
	symbol -= 1
	if symbol < 0 {
		return -1
	}*/
	meta := PageRank(r, r)
	metas := make([]Sample, len(meta))
	for i, v := range meta {
		metas[i].Ranks = r.Data[i*r.Cols : (i+1)*r.Cols]
		metas[i].Meta = v
		metas[i].S = samples[i].S
		ranks := metas[i].Ranks[1:]
		length := len(ranks)
		avg := 0.0
		for _, v := range ranks {
			avg += float64(v)
		}
		avg /= float64(length)
		stddev := 0.0
		for _, v := range ranks {
			diff := float64(v) - avg
			stddev += diff * diff
		}
		stddev /= float64(length)
		stddev = math.Sqrt(stddev)
		metas[i].Avg = avg
		metas[i].Stddev = stddev
	}
	sort.Slice(metas, func(i, j int) bool {
		return float64(metas[i].Meta)/metas[i].Stddev > float64(metas[j].Meta)/metas[j].Stddev
	})
	syms := make([]float32, SetSize)
	for i := range metas {
		syms[metas[i].S] += metas[i].Meta / float32(metas[i].Stddev)
	}
	fmt.Println("syms", syms)
	avg := [SetSize]float32{}
	count := [SetSize]float32{}
	for sample := range samples {
		index := samples[sample].S
		ranks := samples[sample].Ranks
		/*a, c := 0.0, 0.0
		for _, v := range ranks[1 : len(ranks)-1] {
			a += v
			c++
		}
		a /= c
		a -= ranks[len(ranks)-1]
		if a < 0 {
			a = -a
		}*/
		avg[index] += ranks[len(ranks)-1]
		count[index]++
	}
	for i := range avg {
		avg[i] /= count[i]
	}
	stddev := [SetSize]float32{}
	for sample := range samples {
		index := samples[sample].S
		ranks := samples[sample].Ranks
		/*a, c := 0.0, 0.0
		for _, v := range ranks[1 : len(ranks)-1] {
			a += v
			c++
		}
		a /= c
		a -= ranks[len(ranks)-1]
		if a < 0 {
			a = -a
		}*/
		diff := avg[index] - ranks[len(ranks)-1]
		stddev[index] += diff * diff
	}
	for i, v := range stddev {
		stddev[i] = float32(math.Sqrt(float64(v)))
	}
	fmt.Println(avg)
	fmt.Println(stddev)
	metric := [SetSize]float32{}
	for i, v := range avg {
		metric[i] = v / stddev[i]
	}
	fmt.Println(metric)
	/*max, sym := 0.0, 0
	for key, value := range avg {
		value /= stddev[key]
		if value > max {
			max, sym = value, key
		}
	}*/
	max, sym := 0.0, 0
	for key, value := range results {
		if value > max {
			max, sym = value, key
		}
	}
	fmt.Println(max, sym)
	//return int(To[opt.Input[symbol]]) + 1
	return sym + 1
}

// Text mode
func Text2(full bool, s int, seed uint32) int {
	sets := Load()
	rng := matrix.Rand(seed)
	opt := sets.GetSingleTrainingData(s)
	fmt.Println(string(opt.Input))
	fmt.Println(string(opt.Output))
	var y *mat.Dense
	for i := 0; i < 32; i++ {
		seed := rng.Uint32()
		if seed == 0 {
			seed = 1
		}
		samples := Search(sets, s, seed)
		input := make([]float64, 0, len(samples[0].Ranks)*len(samples))
		for sample := range samples {
			ranks := samples[sample].Ranks
			for _, rank := range ranks {
				input = append(input, float64(rank))
			}
		}
		x := mat.NewDense(len(samples), len(samples[0].Ranks), input)
		dst := mat.SymDense{}
		stat.CovarianceMatrix(&dst, x, nil)
		if y == nil {
			rr, cc := dst.Dims()
			y = mat.NewDense(rr, cc, make([]float64, rr*cc))
		}
		y.Add(y, &dst)
	}
	fa := mat.Formatted(y, mat.Squeeze())
	fmt.Println(fa)
	rr, cc := y.Dims()
	graph := pagerank.NewGraph()
	for i := 0; i < rr; i++ {
		for j := 0; j < cc; j++ {
			d := y.At(i, j)
			if d > 0 {
				graph.Link(uint32(i), uint32(j), d)
				graph.Link(uint32(j), uint32(i), d)
			}
		}
	}
	ranks := make([]float64, rr)
	graph.Rank(1, 1e-6, func(node uint32, rank float64) {
		ranks[node] = float64(rank)
	})
	fmt.Println("ranks", ranks)
	results := make([]float64, 4)
	counts := make([]float64, 4)
	for i, v := range opt.Input {
		results[To[v]] += ranks[i+1]
		counts[To[v]]++
	}
	for i := range results {
		results[i] /= counts[i]
	}
	fmt.Println("results", results)
	return 0
}
