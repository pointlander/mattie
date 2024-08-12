// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"

	"github.com/pointlander/matrix"
	"github.com/pointlander/matrix/vector"
)

const (
	Reset   = "\033[0m"
	Red     = "\033[31m"
	Green   = "\033[32m"
	Yellow  = "\033[33m"
	Blue    = "\033[34m"
	Magenta = "\033[35m"
	Cyan    = "\033[36m"
	Gray    = "\033[37m"
	White   = "\033[97m"
)

const (
	// Symbols
	Symbols = 10 + 3 + 3 + 5
	// Input is the network input size
	Input = Symbols + 2*7 + 10
	// Width is the width of the markov model
	Width = 3
	// Height is the height of the markov model
	Height = 3
	// Size is the size of the markov model
	Size = Width*Height - 1
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

func softmax(values []float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V matrix.Matrix) matrix.Matrix {
	o := matrix.Matrix{
		Cols: V.Cols,
		Rows: K.Rows,
		Data: make([]float32, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float32, V.Cols), make([]float32, Q.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = vector.Dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = vector.Dot(values, V)
		}
		softmax(outputs)
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// Example is a learning example
type Example struct {
	Input  [][]byte `json:"input"`
	Output [][]byte `json:"output"`
}

// Set is a set of examples
type Set struct {
	Test  []Example `json:"test"`
	Train []Example `json:"train"`
}

// Sets is many sets
type Sets []Set

// Load loads the data
func Load() Sets {
	dirs, err := os.ReadDir("ARC-AGI/data/training/")
	if err != nil {
		panic(err)
	}
	sets := make(Sets, len(dirs))
	for i, dir := range dirs {
		data, err := os.ReadFile("ARC-AGI/data/training/" + dir.Name())
		if err != nil {
			panic(err)
		}
		err = json.Unmarshal(data, &sets[i])
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("loaded", len(sets))
	test, train := 0, 0
	for _, set := range sets {
		test += len(set.Test)
		train += len(set.Train)
	}
	fmt.Println("test", test)
	fmt.Println("train", train)
	max := 0
	for _, set := range sets {
		if len(set.Train) > max {
			max = len(set.Train)
		}
	}
	fmt.Println("max train", max)
	return sets
}

// Pixel is an image pixel
type Pixel struct {
	C uint8
	X int
	Y int
}

// Image is an image
type Image struct {
	W int
	H int
	I []Pixel
}

// Pair is an input output pair
type Pair struct {
	Class  int
	Input  Image
	Output Image
}

// Problem is an optimization problem
type Problem struct {
	Count  int
	Opt    matrix.Matrix
	Input  []Pair
	Output Pair
	Tail   int
}

// Size is the size of the input
func (p Problem) Size() int {
	sum := 0
	for _, input := range p.Input {
		sum += len(input.Input.I) + len(input.Output.I) + 1 + input.Input.H + 1 + input.Output.H
	}
	return sum + len(p.Output.Input.I) + p.Output.Input.H + 1 + 2 + p.Tail
}

// GetSingleTrainingData gets the training data
func (sets Sets) GetSingleTrainingData(tail, s, t int) Problem {
	train, test := make([]Pair, 0, 8), make([]Pair, 0, 8)
	set := sets[s]
	for x, t := range set.Train {
		pair := Pair{
			Class: s,
			Input: Image{
				W: len(t.Input[0]),
				H: len(t.Input),
			},
			Output: Image{
				W: len(t.Output[0]),
				H: len(t.Output),
			},
		}
		for j, v := range t.Input {
			for i := range v {
				pix := v[i]
				if pix == 0 {
					pix = 16 + byte(x)
				}
				pair.Input.I = append(pair.Input.I, Pixel{
					C: pix,
					X: i,
					Y: j,
				})
			}
		}
		for j, v := range t.Output {
			for i := range v {
				pix := v[i]
				if pix == 0 {
					pix = 16 + byte(x)
				}
				pair.Output.I = append(pair.Output.I, Pixel{
					C: pix,
					X: i,
					Y: j,
				})
			}
		}
		train = append(train, pair)
	}
	for _, t := range set.Test {
		pair := Pair{
			Class: s,
			Input: Image{
				W: len(t.Input[0]),
				H: len(t.Input),
			},
			Output: Image{
				W: len(t.Output[0]),
				H: len(t.Output),
			},
		}
		for j, v := range t.Input {
			for i := range v {
				pair.Input.I = append(pair.Input.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		for j, v := range t.Output {
			for i := range v {
				pair.Output.I = append(pair.Output.I, Pixel{
					C: v[i],
					X: i,
					Y: j,
				})
			}
		}
		test = append(test, pair)
	}
	problem := Problem{
		Count: len(train),
		Tail:  tail,
	}
	problem.Input = train
	problem.Output = test[t]
	problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
	index := 0
	for j, pair := range train {
		problem.Opt.Data[index+10] = 1
		index += Input
		for i, p := range pair.Input.I {
			problem.Opt.Data[index+int(p.C)] = 1
			problem.Opt.Data[index+Input-10+j] = 1
			index += Input
			if (i+1)%pair.Input.W == 0 && i+1 != len(pair.Input.I) {
				problem.Opt.Data[index+11] = 1
				index += Input
			}
		}
		problem.Opt.Data[index+12] = 1
		index += Input
		problem.Opt.Data[index+13] = 1
		index += Input
		for i, p := range pair.Output.I {
			problem.Opt.Data[index+int(p.C)] = 1
			problem.Opt.Data[index+Input-10+j] = 1
			index += Input
			if (i+1)%pair.Output.W == 0 && i+1 != len(pair.Output.I) {
				problem.Opt.Data[index+14] = 1
				index += Input
			}
		}
		problem.Opt.Data[index+15] = 1
		index += Input
	}
	problem.Opt.Data[index+10] = 1
	index += Input
	for i, p := range test[t].Input.I {
		problem.Opt.Data[index+int(p.C)] = 1
		for j := range train {
			problem.Opt.Data[index+Input-10+j] = 1
		}
		index += Input
		if (i+1)%test[t].Input.W == 0 && i+1 != len(test[t].Input.I) {
			problem.Opt.Data[index+11] = 1
			index += Input
		}
	}
	problem.Opt.Data[index+12] = 1
	index += Input
	problem.Opt.Data[index+13] = 1
	index += Input
	for j := 0; j < 1+tail; j++ {
		for j := range train {
			problem.Opt.Data[index+Input-10+j] = 1
		}
		index += Input
	}
	return problem
}

// Model model is the random matrix model
type Model struct {
	Query    matrix.CompressedRandomMatrix
	Key      matrix.CompressedRandomMatrix
	Value    matrix.CompressedRandomMatrix
	Solution matrix.CompressedRandomMatrix
	Order    matrix.CompressedRandomMatrix
}

// Sample is a sample
type Sample struct {
	Query matrix.CompressedGenerator
	Key   matrix.CompressedGenerator
	Value matrix.CompressedGenerator
	S     int
	Order matrix.CompressedGenerator
	Cost  float64
	Grid  [][]int
}

// Stat is a statistic
type Stat struct {
	Count      float64
	Sum        float64
	SumSquared float64
}

// State is a markov state
type State [Size]byte

// Text mode
func Text() {
	sets := Load()
	const (
		SampleSets = 4000
		Samples    = SampleSets * Symbols
	)
	type Result struct {
		Context int
		Symbol  int
		Score   float64
	}
	var search func(context int, seed uint32, suffix []byte, depth int, results chan Result)
	search = func(context int, seed uint32, suffix []byte, depth int, results chan Result) {
		depth--
		opt := sets.GetSingleTrainingData(len(suffix), 0, 0)
		model := Model{
			Query: matrix.NewCompressedRandomMatrix(Input, Input),
			Key:   matrix.NewCompressedRandomMatrix(Input, Input),
			Value: matrix.NewCompressedRandomMatrix(Input, Input),
			Order: matrix.NewCompressedRandomMatrix(7, opt.Size()),
		}
		stats := make([]float64, Symbols)
		samples := make([]Sample, Samples)
		rng := matrix.Rand(seed)
		for i := 0; i < SampleSets; i++ {
			for j := 0; j < Symbols; j++ {
				query := model.Query.Sample(&rng)
				key := model.Key.Sample(&rng)
				value := model.Value.Sample(&rng)
				order := model.Order.Sample(&rng)
				samples[i*Symbols+j].Query = query
				samples[i*Symbols+j].Key = key
				samples[i*Symbols+j].Value = value
				samples[i*Symbols+j].Order = order
				samples[i*Symbols+j].S = j
			}
		}
		done := make(chan bool, 8)
		process := func(sample *Sample) {
			opt := sets.GetSingleTrainingData(len(suffix), 0, 0)
			sum := 0.0
			order := sample.Order.Sample()
			a, b := 0, 1
			for j := 0; j < opt.Opt.Rows; j++ {
				x, y := (j+a)%opt.Opt.Rows, (j+b)%opt.Opt.Rows
				copy(opt.Opt.Data[j*Input+Symbols:j*Input+Symbols+7], order.Data[x*7:(x+1)*7])
				copy(opt.Opt.Data[j*Input+Symbols+7:j*Input+Symbols+2*7], order.Data[(y)*7:(y+1)*7])
				a, b = b, a
			}
			index := 0
			for i := len(suffix) + 1; i > 1; i-- {
				opt.Opt.Data[Input*(opt.Size()-i)+int(suffix[index])] = 1
				index++
			}
			params := opt.Opt.Data[Input*(opt.Size()-1):]
			params[sample.S] = 1
			/*out := matrix.SelfAttention(
			sample.Query.MulT(opt.Opt),
			sample.Key.MulT(opt.Opt),
			sample.Value.MulT(opt.Opt))*/
			query := sample.Query.Sample()
			key := sample.Key.Sample()
			value := sample.Value.Sample()
			entropy := matrix.SelfEntropy(
				query.MulT(opt.Opt),
				key.MulT(opt.Opt),
				value.MulT(opt.Opt))
			for _, value := range entropy {
				sum += float64(value)
			}
			/*for j := 0; j < out.Rows; j++ {
				for k := 0; k < out.Cols; k++ {
					diff := out.Data[j*out.Cols+k] - opt.Opt.Data[j*out.Cols+k]
					sum += float64(diff*diff)
				}
			}*/
			sample.Cost = sum
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

		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Cost < samples[j].Cost
		})
		/*avg, vr := 0.0, 0.0
		for i := 0; i < len(samples); i++ {
			avg += samples[i].Cost
		}
		avg /= float64(len(samples))
		for i := 0; i < len(samples); i++ {
			diff := samples[i].Cost - avg
			vr += diff * diff
		}
		vr /= float64(len(samples))
		type Cut struct {
			Reduction float64
			Index     int
		}
		cuts := make(chan Cut, 8)
		mvr := func(i int) {
			avga, avgb := 0.0, 0.0
			vara, varb := 0.0, 0.0
			for j := 0; j < i; j++ {
				avga += samples[j].Cost
			}
			avga /= float64(i)
			for j := 0; j < i; j++ {
				diff := samples[j].Cost - avga
				vara += diff * diff
			}
			vara /= float64(i)
			for j := i; j < len(samples); j++ {
				avgb += samples[j].Cost
			}
			avgb /= float64(len(samples) - i)
			for j := i; j < len(samples); j++ {
				diff := samples[j].Cost - avgb
				varb += diff * diff
			}
			varb /= float64(len(samples) - i)
			reduction := vr - (vara + varb)
			cuts <- Cut{
				Reduction: reduction,
				Index:     i,
			}
		}
		maxReduction, cut := 0.0, 0
		flight, index, cpus = 0, 1, runtime.NumCPU()
		for flight < cpus && index < len(samples)-1 {
			go mvr(index)
			index++
			flight++
		}
		for index < len(samples)-1 {
			result := <-cuts
			if result.Reduction > maxReduction {
				maxReduction, cut = result.Reduction, result.Index
			}
			flight--

			go mvr(index)
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			result := <-cuts
			if result.Reduction > maxReduction {
				maxReduction, cut = result.Reduction, result.Index
			}
		}*/

		acc := [Symbols]int{}
		factor := [Symbols]int{}
		for sample := range samples {
			index := samples[sample].S
			acc[index]++
			scale := sample - factor[index]
			factor[index] = sample
			if scale == 0 {
				scale = 1
			}
			max, sym := 0.0, 0
			for key, value := range acc {
				if float64(value) > max {
					max, sym = float64(value), key
				}
			}
			stats[sym] += 1 / float64(scale)
		}
		fmt.Println(stats)

		max, index := 0.0, 0
		if depth > 0 {
			results := make(chan Result, Symbols)
			for i := range stats[:16] {
				cp := make([]byte, len(suffix))
				copy(cp, suffix)
				s := append(cp, byte(i))
				seed := rng.Uint32() + 1
				if seed == 0 {
					seed = 1
				}
				go search(i, seed, s, depth, results)
			}
			count := 0
			for result := range results {
				if score := result.Score + stats[result.Context]; score > max {
					max, index = score, result.Context
				}
				count++
				if count == Symbols {
					break
				}
			}
		} else {
			for i, stat := range stats[:16] {
				if stat > max {
					max, index = stat, i
				}
			}
		}
		results <- Result{
			Context: context,
			Symbol:  index,
			Score:   max,
		}
	}
	results := make(chan Result, Symbols)
	search(0, 1, []byte{}, 1, results)
	result := <-results
	fmt.Println(result.Symbol, result.Score)
	symbols := []byte{byte(result.Symbol)}
	search(0, 2, symbols, 1, results)
	result = <-results
	fmt.Println(result.Symbol, result.Score)
	symbols = append(symbols, byte(result.Symbol))
	search(0, 3, symbols, 1, results)
	result = <-results
	fmt.Println(result.Symbol, result.Score)
}
