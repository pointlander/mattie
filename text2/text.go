// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text2

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"

	"github.com/pointlander/matrix"
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
	Symbols = 10 + 3 + 3
	// Input is the network input size
	Input = Symbols + 2*7
	// Width is the width of the markov model
	Width = 3
	// Height is the height of the markov model
	Height = 3
	// Size is the size of the markov model
	Size = Width*Height - 1
)

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
	for _, t := range set.Train {
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
	}
	problem.Input = train
	problem.Output = test[t]
	problem.Opt = matrix.NewZeroMatrix(Input, problem.Size())
	index := 0
	for _, pair := range train {
		problem.Opt.Data[index+10] = 1
		index += Input
		for i, p := range pair.Input.I {
			problem.Opt.Data[index+int(p.C)] = 1
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
	return problem
}

// Model model is the random matrix model
type Model struct {
	Query    matrix.RandomMatrix
	Key      matrix.RandomMatrix
	Value    matrix.RandomMatrix
	Solution matrix.RandomMatrix
	Order    matrix.RandomMatrix
}

// Sample is a sample
type Sample struct {
	Query matrix.Generator
	Key   matrix.Generator
	Value matrix.Generator
	S     int
	Order matrix.Generator
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

// Text2 mode
func Text2() {
	sets := Load()

	type Result struct {
		Symbol int
		Score  int
	}
	var search func(suffix []byte, depth int, results chan Result)
	search = func(suffix []byte, depth int, results chan Result) {
		depth--
		opt := sets.GetSingleTrainingData(len(suffix), 0, 0)
		model := Model{
			Query: matrix.NewRandomMatrix(Input, Input),
			Key:   matrix.NewRandomMatrix(Input, Input),
			Value: matrix.NewRandomMatrix(Input, Input),
			Order: matrix.NewRandomMatrix(7, opt.Size()),
		}
		stats := make([]int, Symbols)
		samples := make([]Sample, 100*Symbols)
		seed := uint32(1)
		seed += uint32(depth)
		for _, s := range suffix {
			seed += uint32(s)
		}
		rng := matrix.Rand(seed)
		for i := range samples {
			samples[i].Query = model.Query.Sample(&rng)
			samples[i].Key = model.Key.Sample(&rng)
			samples[i].Value = model.Value.Sample(&rng)
			samples[i].Order = model.Order.Sample(&rng)
			samples[i].S = i % Symbols
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
		avg, vr := 0.0, 0.0
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
		}

		samples = samples[:cut]
		for sample := range samples {
			index := samples[sample].S
			stats[index]++
		}

		max, index := 0, 0
		if depth > 0 {
			results := make(chan Result, Symbols)
			for i := range stats {
				s := append(suffix, byte(i))
				go search(s, depth, results)
			}
			count := 0
			for result := range results {
				if score := result.Score + stats[result.Symbol]; score > max {
					max, index = score, result.Symbol
				}
				count++
				if count == Symbols {
					break
				}
			}
		} else {
			for i, stat := range stats {
				if stat > max {
					max, index = stat, i
				}
			}
		}
		results <- Result{
			Symbol: index,
			Score:  max,
		}
	}
	results := make(chan Result, Symbols)
	search([]byte{}, 2, results)
	result := <-results
	fmt.Println(result.Symbol, result.Score)
	symbols := []byte{byte(result.Symbol)}
	search(symbols, 2, results)
	result = <-results
	fmt.Println(result.Symbol, result.Score)
	symbols = append(symbols, byte(result.Symbol))
	search(symbols, 2, results)
	result = <-results
	fmt.Println(result.Symbol, result.Score)
}
