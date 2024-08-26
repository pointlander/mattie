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

	"github.com/pointlander/matrix"
	"github.com/pointlander/matrix/vector"

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
	// SetSize is the size of a symbol set
	SetSize = 4
	// SampleSets is the number of samples per set
	SampleSets = 100
	// Samples is the number of samplee
	Samples = SampleSets * SetSize
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
func PageRank(Q, K matrix.Matrix) float64 {
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
			graph.Link(uint32(i), uint32(j), d*d)
		}
	}
	result := 0.0
	graph.Rank(0.85, 1e-6, func(node uint32, rank float64) {
		if node == uint32(K.Rows-1) {
			result = rank
		}
	})
	return result
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
		Count:  len(txt) + 1,
	}
	problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
	return problem
}

// Model model is the random matrix model
type Model struct {
	Query  matrix.CompressedRandomMatrix
	Key    matrix.CompressedRandomMatrix
	Order  matrix.CompressedRandomMatrix
	Symbol matrix.CompressedRandomMatrix
}

// Sample is a sample
type Sample struct {
	Rng    *matrix.Rand
	Query  matrix.CompressedGenerator
	Key    matrix.CompressedGenerator
	Order  matrix.CompressedGenerator
	Symbol matrix.CompressedGenerator
	S      int
	Cost   float64
	Corr   float64
	Mult   float64
}

// Search searches for a symbol
func Search(sets Sets, s int, seed uint32) []Sample {
	opt := sets.GetSingleTrainingData(s)
	model := Model{
		Query:  matrix.NewCompressedRandomMatrix(Input, Input),
		Key:    matrix.NewCompressedRandomMatrix(Input, Input),
		Order:  matrix.NewCompressedRandomMatrix(Size, opt.Size()),
		Symbol: matrix.NewCompressedRandomMatrix(Size, Symbols),
	}
	samples := make([]Sample, Samples)
	rng := matrix.Rand(seed)
	for i := 0; i < SampleSets; i++ {
		for j := 0; j < SetSize; j++ {
			query := model.Query.Sample(&rng)
			key := model.Key.Sample(&rng)
			//rngOrder := matrix.Rand(1)
			order := model.Order.Sample(&rng)
			//rngSymbol := matrix.Rand(1)
			symbol := model.Symbol.Sample(&rng)
			seed := rng.Uint32()
			if seed == 0 {
				seed += 1
			}
			Rng := matrix.Rand(seed)
			samples[i*SetSize+j].Rng = &Rng
			samples[i*SetSize+j].Query = query
			samples[i*SetSize+j].Key = key
			samples[i*SetSize+j].Order = order
			samples[i*SetSize+j].Symbol = symbol
			samples[i*SetSize+j].S = j
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
		jj := opt.Opt.Rows - 1
		for j := 0; j < jj; j++ {
			x, y := (j+a)%opt.Opt.Rows, (j+b)%opt.Opt.Rows
			copy(opt.Opt.Data[j*Input+Size:j*Input+Size+Size],
				order.Data[x*Size:(x+1)*Size])
			copy(opt.Opt.Data[j*Input+Size+Size:j*Input+Size+2*Size],
				order.Data[(y)*Size:(y+1)*Size])
			a, b = b, a
		}
		if x := jj + a; x < opt.Opt.Rows {
			copy(opt.Opt.Data[jj*Input+Size:jj*Input+Size+Size],
				order.Data[x*Size:(x+1)*Size])
		}
		if y := jj + b; y < opt.Opt.Rows {
			copy(opt.Opt.Data[jj*Input+Size+Size:jj*Input+Size+2*Size],
				order.Data[(y)*Size:(y+1)*Size])
		}
		syms := sample.Symbol.Sample()
		index := Input
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
		query := sample.Query.Sparse()
		key := sample.Key.Sparse()
		mult := query.MulT(key)
		sum := 0.0
		for i := range mult.Data {
			sum += math.Abs(float64(mult.Data[i]))
		}
		sample.Mult = sum
		found := 0.0
		/*for {
			corr := 0.0
			mquery, mkey := 0.0, 0.0
			for _, v := range query.Data {
				mquery += float64(v)
			}
			mquery /= float64(len(query.Data))
			for _, v := range key.Data {
				mkey += float64(v)
			}
			mkey /= float64(len(query.Data))
			stddevquery := 0.0
			stddevkey := 0.0

			for i, v := range query.Data {
				diffquery := float64(v) - mquery
				diffkey := float64(key.Data[i]) - mkey
				stddevquery += diffquery * diffquery
				stddevkey += diffkey * diffkey
				corr += diffquery * diffkey
			}
			stddevquery = math.Sqrt(stddevquery)
			stddevkey = math.Sqrt(stddevkey)
			corr = (corr / float64(len(query.Data))) / (stddevquery * stddevkey)
			if corr > 1e-10 || corr < -1e-10 {
				found = corr
				break
			}
			query = sample.Query.Sample()
			key = sample.Key.Sample()
		}*/
		sample.Corr = found
		//query = query.H(c)
		//key = key.H(c)
		q := query.MulT(opt.Opt)
		k := key.MulT(opt.Opt)
		sample.Cost = PageRank(q, k)
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
func Text(full bool, s int, ss uint32) int {
	sets := Load()
	opt := sets.GetSingleTrainingData(s)
	fmt.Println(string(opt.Input))
	fmt.Println(string(opt.Output))
	samples := []Sample{}
	rng := matrix.Rand(ss)
	for i := 1; i < 64; i++ {
		seed := rng.Uint32()
		if seed == 0 {
			seed = 1
		}
		result := Search(sets, s, seed)
		samples = append(samples, result...)
	}
	avg := [SetSize]float64{}
	count := [SetSize]float64{}
	corr := 0.0
	mult := 0.0
	for sample := range samples {
		index := samples[sample].S
		avg[index] += samples[sample].Cost
		count[index]++
		corr += samples[sample].Corr
		mult += samples[sample].Mult
	}
	for i := range avg {
		avg[i] /= count[i]
	}
	stddev := [SetSize]float64{}
	for sample := range samples {
		index := samples[sample].S
		diff := avg[index] - samples[sample].Cost
		stddev[index] += diff * diff
	}
	for i, v := range stddev {
		stddev[i] = math.Sqrt(v)
	}
	fmt.Println(avg)
	fmt.Println(stddev)
	metric := [SetSize]float64{}
	for i, v := range avg {
		metric[i] = v / stddev[i]
	}
	fmt.Println(metric)
	fmt.Println(corr)
	fmt.Println(mult)
	max, sym := 0.0, 0
	for key, value := range avg {
		value /= stddev[key]
		if value > max {
			max, sym = value, key
		}
	}
	return sym + 1
}
