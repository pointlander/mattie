// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package random

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"

	"github.com/pointlander/matrix"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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
	Symbols = 11
	// Input is the network input size
	Input = Symbols + 2*7 + 1
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

// Load loads the data
func Load() []Set {
	dirs, err := os.ReadDir("ARC-AGI/data/training/")
	if err != nil {
		panic(err)
	}
	sets := make([]Set, len(dirs))
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

// Opt is an optimization
type Opt struct {
	Opt    matrix.Matrix
	Input  Pair
	Output Pair
}

// TargetOffset is the target offset
func (o Opt) TargetOffset() int {
	return len(o.Input.Input.I) + len(o.Input.Output.I) + len(o.Output.Input.I) + 3
}

// TargetSize is the size of the target
func (o Opt) TargetSize() int {
	return len(o.Output.Output.I)
}

// GetTrainingData gets the training data
func GetTrainingData(sets []Set, s, t int) (opt []Opt, w [Symbols]int) {
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
	opt = make([]Opt, len(train))
	for i := range opt {
		opt[i].Input = train[i]
		opt[i].Output = test[t]
		opt[i].Opt = matrix.NewZeroMatrix(Input, opt[i].TargetOffset()+opt[i].TargetSize())
	}
	for i, pair := range train {
		index := 0
		for _, p := range pair.Input.I {
			w[p.C]++
			opt[i].Opt.Data[index+int(p.C)] = 1
			index += Input
		}
		w[10]++
		opt[i].Opt.Data[index+10] = 1
		index += Input
		for _, p := range pair.Output.I {
			w[p.C]++
			opt[i].Opt.Data[index+int(p.C)] = 1
			opt[i].Opt.Data[index+Symbols+2*7] = 1
			index += Input
		}
		w[10]++
		opt[i].Opt.Data[index+10] = 1
		index += Input

		for _, p := range test[t].Input.I {
			w[p.C]++
			opt[i].Opt.Data[index+int(p.C)] = 1
			index += Input
		}
		w[10]++
		opt[i].Opt.Data[index+10] = 1
		index += Input
	}
	return opt, w
}

// OptSingle is an optimization
type OptSingle struct {
	Count  int
	Opt    matrix.Matrix
	Input  Pair
	Output Pair
}

// TargetOffset is the target offset
func (o OptSingle) TargetOffset() int {
	return o.Count*(len(o.Input.Input.I)+len(o.Input.Output.I)+2) + len(o.Output.Input.I) + 1
}

// TargetSize is the size of the target
func (o OptSingle) TargetSize() int {
	return len(o.Output.Output.I)
}

// GetSingleTrainingData gets the training data
func GetSingleTrainingData(sets []Set, s, t int) (opt []OptSingle, w [Symbols]int) {
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
	opt = make([]OptSingle, 1)
	opt[0].Count = len(train)
	for i := range opt {
		opt[i].Input = train[i]
		opt[i].Output = test[t]
		opt[i].Opt = matrix.NewZeroMatrix(Input, opt[i].TargetOffset()+opt[i].TargetSize())
	}
	index := 0
	for _, pair := range train {
		for _, p := range pair.Input.I {
			w[p.C]++
			opt[0].Opt.Data[index+int(p.C)] = 1
			index += Input
		}
		w[10]++
		opt[0].Opt.Data[index+10] = 1
		index += Input
		for _, p := range pair.Output.I {
			w[p.C]++
			opt[0].Opt.Data[index+int(p.C)] = 1
			opt[0].Opt.Data[index+Symbols+2*7] = 1
			index += Input
		}
		w[10]++
		opt[0].Opt.Data[index+10] = 1
		index += Input
	}
	for _, p := range test[t].Input.I {
		w[p.C]++
		opt[0].Opt.Data[index+int(p.C)] = 1
		index += Input
	}
	w[10]++
	opt[0].Opt.Data[index+10] = 1
	index += Input
	return opt, w
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
	Query    matrix.Generator
	Key      matrix.Generator
	Value    matrix.Generator
	Solution matrix.Generator
	Order    matrix.Generator
	Cost     float64
	Grid     [][]int
}

// Stat is a statistic
type Stat struct {
	Count      float64
	Sum        float64
	SumSquared float64
}

// State is a markov state
type State [Size]byte

// Original mode
func Random() {
	rng := matrix.Rand(1)
	sets := Load()
	_ = sets
	//opts, ww := GetTrainingData(sets, 0, 0)
	opts, ww := GetSingleTrainingData(sets, 0, 0)
	model := Model{
		Query:    matrix.NewRandomMatrix(Input, Input),
		Key:      matrix.NewRandomMatrix(Input, Input),
		Value:    matrix.NewRandomMatrix(Input, Input),
		Solution: matrix.NewRandomMatrix(Symbols, opts[0].TargetSize()),
		Order:    matrix.NewRandomMatrix(7, opts[0].TargetOffset()+opts[0].TargetSize()),
	}
	votes := make([][]int, opts[0].Output.Output.H*opts[0].Output.Output.W)
	stats := make([][]Stat, opts[0].Output.Output.H*opts[0].Output.Output.W)
	for v := range votes {
		votes[v] = make([]int, Symbols)
		stats[v] = make([]Stat, Symbols)
	}
	markov, state := make(map[State][Symbols]int), State{}
	var auto, acc plotter.Values
	grids := make([][][]byte, 0, 8)
	//for i := 0; i < 4*1024; i++ {
	//fmt.Println(i)
	samples := make([]Sample, 4*1024)
	maxReduction, cut := 0.0, 0
	{
		for i := range samples {
			samples[i].Query = model.Query.Sample(&rng)
			samples[i].Key = model.Key.Sample(&rng)
			samples[i].Value = model.Value.Sample(&rng)
			samples[i].Solution = model.Solution.Sample(&rng)
			samples[i].Order = model.Order.Sample(&rng)
		}

		done := make(chan bool, 8)
		process := func(sample *Sample) {
			//opts, _ := GetTrainingData(sets, 0, 0)
			opts, _ := GetSingleTrainingData(sets, 0, 0)
			sum := 0.0
			for _, opt := range opts {
				/*input := matrix.NewZeroMatrix(inputSize, 1)
				for i, value := range opt.Output.Input.I {
					input.Data[i*10+int(value.C)] = 1
				}
				output := sample.W2.MulT(sample.W1.MulT(input).Add(sample.B1).Sigmoid()).Add(sample.B2).Sigmoid()*/
				order := sample.Order.Sample()
				a, b := 0, 1
				for j := 0; j < opt.Opt.Rows; j++ {
					x, y := (j+a)%opt.Opt.Rows, (j+b)%opt.Opt.Rows
					copy(opt.Opt.Data[j*Input+Symbols:j*Input+Symbols+7], order.Data[x*7:(x+1)*7])
					copy(opt.Opt.Data[j*Input+Symbols+7:j*Input+Symbols+2*7], order.Data[(y)*7:(y+1)*7])
					a, b = b, a
				}
				solution := sample.Solution.Sample()
				params := opt.Opt.Data[Input*opt.TargetOffset():]
				for j := 0; j < solution.Rows; j++ {
					max, index := 0.0, 0
					for k := 0; k < solution.Cols; k++ {
						//value := float64(output.Data[j*10+k])
						value := float64(solution.Data[j*solution.Cols+k])
						if value < 0 {
							value = -value
						}
						if value > max {
							max, index = value, k
						}
					}
					params[j*Input+index] = 1
					params[j*Input+Symbols+2*7] = 1
				}
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
			}
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
		for i := 1; i < len(samples)-1; i++ {
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
			if reduction > maxReduction {
				maxReduction, cut = reduction, i
			}
		}
		samples = samples[:cut]
		for sample := range samples {
			//fmt.Println(samples[0].Cost)
			h, w := opts[0].Output.Output.H, opts[0].Output.Output.W
			grid := make([][]byte, h)
			for j := range grid {
				grid[j] = make([]byte, w)
			}
			/*input := matrix.NewZeroMatrix(inputSize, 1)
			for i, value := range opts[0].Output.Input.I {
				input.Data[i*10+int(value.C)] = 1
			}
			output := samples[0].W2.MulT(samples[0].W1.MulT(input).Add(samples[0].B1).Sigmoid()).Add(samples[0].B2).Sigmoid()*/
			solution := samples[sample].Solution.Sample()
			for j := 0; j < solution.Rows; j++ {
				max, index := 0.0, 0
				for k := 0; k < solution.Cols; k++ {
					//value := float64(output.Data[j*10+k])
					value := float64(solution.Data[j*solution.Cols+k])
					if value < 0 {
						value = -value
					}
					if value > max {
						max, index = value, k
					}
				}
				stats[j][index].Count++
				stats[j][index].Sum += max
				stats[j][index].SumSquared += max * max
				votes[j][index]++
				grid[j/h][j%w] = byte(index)
			}
			grids = append(grids, grid)
			correct, count := 0.0, 0.0
			for j := 0; j < h; j++ {
				for i := 0; i < w; i++ {
					context := 0
					state = State{}
					for x := -Width / 2; x < Width/2; x++ {
						for y := -Height / 2; y < Height/2; y++ {
							if x == 0 && y == 0 {
								continue
							}
							xx, yy := i+x, j+y
							if xx < 0 || yy < 0 || xx >= w || yy >= h {
								state[context] = byte(Symbols & 0xFF)
								context++
								continue
							}
							state[context] = byte(grid[yy][xx] & 0xFF)
							context++
						}
					}
					s := markov[state]
					s[grid[j][i]]++
					markov[state] = s
					count++
					value := opts[0].Output.Output.I[j*w+i].C
					if value == grid[j][i] {
						fmt.Printf("* ")
						correct++
						continue
					}
					fmt.Printf("%d ", grid[j][i])
				}
				fmt.Println()
			}
			fmt.Println(correct / count)
			auto = append(auto, samples[0].Cost)
			acc = append(acc, correct/count)
		}

		//for i := 0; i < model.Solution.Rows; i++ {
		//	for j := 0; j < model.Solution.Cols; j++ {
		//avg := stats[i][j].Sum / stats[i][j].Count
		//model.Solution.Data[i*model.Solution.Cols+j].Mean = avg
		//model.Solution.Data[i*model.Solution.Cols+j].StdDev =
		//	math.Sqrt(stats[i][j].SumSquared/stats[i][j].Count - avg*avg)
		//		model.Solution.Data[i*model.Solution.Cols+j].StdDev = float64(votes[i][j])
		//	}
		//}
	}

	h, w := opts[0].Output.Output.H, opts[0].Output.Output.W
	grid := make([][]byte, h)
	for j := range grid {
		grid[j] = make([]byte, w)
	}
	counts := make([]int, Symbols)
	for i := range votes {
		max, index := 0, 0
		for j, value := range votes[i] {
			counts[j] += value
			if value > max {
				max, index = value, j
			}
		}
		grid[i/h][i%w] = byte(index)
		fmt.Printf("%d ", index)
		if (i+1)%w == 0 {
			fmt.Println()
		}
	}
	for i, count := range counts {
		if ww[i] == 0 {
			fmt.Printf("0 ")
			continue
		}
		fmt.Printf("%d ", count/ww[i])
	}
	fmt.Println()
	fmt.Println()
	correct, count := 0.0, 0.0
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			count++
			value := opts[0].Output.Output.I[j*w+i].C
			if value == grid[j][i] {
				fmt.Printf("* ")
				correct++
				continue
			}
			fmt.Printf("%d ", grid[j][i])
		}
		fmt.Println()
	}
	fmt.Println()
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			value := int(opts[0].Output.Output.I[j*w+i].C)
			fmt.Printf("%d ", value)
		}
		fmt.Println()
	}
	for i := 0; i < model.Solution.Rows; i++ {
		for j := 0; j < model.Solution.Cols; j++ {
			avg := stats[i][j].Sum / stats[i][j].Count
			stddev := math.Sqrt(stats[i][j].SumSquared/stats[i][j].Count - avg*avg)
			fmt.Printf("%f ", avg/stddev)
		}
		fmt.Println()
	}
	type Context struct {
		State        State
		Distribution [Symbols]int
	}
	contexts := make([]Context, 0, len(markov))
	fmt.Println()
	for key, value := range markov {
		contexts = append(contexts, Context{
			State:        key,
			Distribution: value,
		})
	}
	sort.Slice(contexts, func(i, j int) bool {
		for k := range contexts[i].State {
			if contexts[i].State[k] < contexts[j].State[k] {
				return true
			} else if contexts[i].State[k] > contexts[j].State[k] {
				return false
			}
		}
		return false
	})
	for _, value := range contexts {
		fmt.Println(value.State, value.Distribution)
	}
	done := make(chan bool, 8)
	process := func(sample *matrix.Sample) {
		x1 := sample.Vars[0][0].Sample()
		y1 := sample.Vars[0][1].Sample()
		z1 := sample.Vars[0][2].Sample()
		weights := x1.Add(y1.H(z1))
		grid := make([][]int, h)
		for j := range grid {
			grid[j] = make([]int, w)
		}
		for j := 0; j < h; j++ {
			for i := 0; i < w; i++ {
				max, index := float32(0), 0
				for k := 0; k < Symbols; k++ {
					value := weights.Data[(j*w+i)*Symbols+k]
					if value < 0 {
						value = -value
					}
					if value > max {
						max, index = value, k
					}
				}
				grid[j][i] = index
			}
		}
		sum := 0.0
		for j := 0; j < h; j++ {
			for i := 0; i < w; i++ {
				context := 0
				state = State{}
				for x := -Width / 2; x < Width/2; x++ {
					for y := -Height / 2; y < Height/2; y++ {
						if x == 0 && y == 0 {
							continue
						}
						xx, yy := i+x, j+y
						if xx < 0 || yy < 0 || xx >= w || yy >= h {
							state[context] = byte(Symbols & 0xFF)
							context++
							continue
						}
						state[context] = byte(grid[yy][xx] & 0xFF)
						context++
					}
				}
				s := markov[state]
				sum += float64(s[grid[j][i]])
			}
		}
		sample.Cost = -sum
		done <- true
	}
	optimizer := matrix.NewOptimizer(&rng, 9, .1, 1, func(samples []matrix.Sample, x ...matrix.Matrix) {
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for flight < cpus && index < len(samples) {
			go process(&samples[index])
			index++
			flight++
		}
		for index < len(samples) {
			<-done
			flight--
			fmt.Printf(".")

			go process(&samples[index])
			index++
			flight++
		}
		for i := 0; i < flight; i++ {
			<-done
			fmt.Printf(".")
		}
		fmt.Printf("\n")
	}, matrix.NewCoord(Symbols, opts[0].TargetSize()))
	var sample matrix.Sample
	for i := 0; i < 33; i++ {
		sample = optimizer.Iterate()
		fmt.Println(i, sample.Cost)
	}
	x1 := sample.Vars[0][0].Sample()
	y1 := sample.Vars[0][1].Sample()
	z1 := sample.Vars[0][2].Sample()
	weights := x1.Add(y1.H(z1))
	grid = make([][]byte, h)
	for j := range grid {
		grid[j] = make([]byte, w)
	}
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			max, index := float32(0.0), 0
			for k := 0; k < Symbols; k++ {
				value := weights.Data[(j*w+i)*Symbols+k]
				if value < 0 {
					value = -value
				}
				if value > max {
					max, index = value, k
				}
			}
			grid[j][i] = byte(index)
		}
	}
	correct3, count3 := 0.0, 0.0
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			count3++
			value := opts[0].Output.Output.I[j*w+i].C
			if value == grid[j][i] {
				correct3++
				fmt.Printf(Blue+"%d "+Reset, grid[j][i])
				continue
			}
			fmt.Printf("%d ", grid[j][i])
		}
		fmt.Println()
	}
	fmt.Println()

	var grid2 [][]byte
	max := 0.0
	for _, grid := range grids {
		sum := 0.0
		for j := 0; j < h; j++ {
			for i := 0; i < w; i++ {
				context := 0
				state = State{}
				for x := -Width / 2; x < Width/2; x++ {
					for y := -Height / 2; y < Height/2; y++ {
						if x == 0 && y == 0 {
							continue
						}
						xx, yy := i+x, j+y
						if xx < 0 || yy < 0 || xx >= w || yy >= h {
							state[context] = byte(Symbols & 0xFF)
							context++
							continue
						}
						state[context] = byte(grid[yy][xx] & 0xFF)
						context++
					}
				}
				s := markov[state]
				sum += float64(s[grid[j][i]])
			}
		}
		if sum > max {
			max, grid2 = sum, grid
		}
	}
	correct2, count2 := 0.0, 0.0
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			count2++
			value := opts[0].Output.Output.I[j*w+i].C
			if value == grid2[j][i] {
				correct2++
				fmt.Printf(Blue+"%d "+Reset, grid2[j][i])
				continue
			}
			fmt.Printf("%d ", grid2[j][i])
		}
		fmt.Println()
	}
	fmt.Println()
	fmt.Println(correct / count)
	fmt.Println(correct3 / count3)
	fmt.Println(correct2 / count2)
	fmt.Println(maxReduction, cut)

	p := plot.New()
	p.Title.Text = "acc histogram plot"

	hist, err := plotter.NewHist(acc, 10)
	if err != nil {
		panic(err)
	}
	p.Add(hist)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "acc_histogram.png")
	if err != nil {
		panic(err)
	}

	p = plot.New()
	p.Title.Text = "auto histogram plot"

	hist, err = plotter.NewHist(auto, 10)
	if err != nil {
		panic(err)
	}
	p.Add(hist)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "auto_histogram.png")
	if err != nil {
		panic(err)
	}

	x, y, xy, xx, yy := 0.0, 0.0, 0.0, 0.0, 0.0
	for i, X := range acc {
		Y := auto[i]
		x += X
		y += Y
		xy += X * Y
		xx += X * X
		yy += Y * Y
	}
	length := float64(len(acc))
	x /= length
	y /= length
	xy /= length
	xx /= length
	yy /= length
	corr := (xy - x*y) / (math.Sqrt(xx-x*x) * math.Sqrt(yy-y*y))
	fmt.Println("corr", corr)
}
