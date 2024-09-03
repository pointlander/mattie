// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"

	"github.com/pointlander/matrix"
	"github.com/pointlander/mattie/original"
	"github.com/pointlander/mattie/random"
	"github.com/pointlander/mattie/text"
	"github.com/pointlander/mattie/text2"
	"github.com/pointlander/mattie/txt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var (
	// FlagOriginal original mode
	FlagOriginal = flag.Bool("original", false, "original mode")
	// RandomMode random mode
	FlagRandom = flag.Bool("random", false, "random mode")
	// TextMode is text mode
	FlagText = flag.Bool("text", false, "text mode")
	// Text2Mode is text 2 mode
	FlagText2 = flag.Bool("text2", false, "text 2 mode")
	// Txt is actual text mode
	FlagTxt = flag.Bool("txt", false, "true txt mode")
	// FlagFull computes the full string
	FlagFull = flag.Bool("full", false, "compute the full string")
	// FlagBrute brute mode
	FlagBrute = flag.Bool("brute", false, "brute mode")
	// FlagFinesse fin mode
	FlagFinesse = flag.Bool("finesse", false, "finesse mode")
)

func main() {
	flag.Parse()

	if *FlagOriginal {
		original.Original()
		return
	} else if *FlagRandom {
		random.Random()
		return
	} else if *FlagText {
		text.Text()
		return
	} else if *FlagText2 {
		text2.Text2()
		return
	} else if *FlagTxt {
		var seed uint32 = 0
	outer:
		for {
			seed++
			fmt.Println(seed)
			for i := 1; i < 5; i++ {
				result := txt.Text(false, i, seed)
				if i != result {
					fmt.Printf("fail %d!=%d\n", i, result)
					continue outer
				}
			}
			result := txt.Text(false, 5, seed)
			if result != 4 {
				fmt.Printf("fail 4!=%d\n", result)
				continue outer
			}
			break
		}
		fmt.Println(seed)
		return
	} else if *FlagBrute {
		histogram := [5][4]int{}
		for h := range histogram {
			points := make(plotter.XYs, 0, 8)
			rng := matrix.Rand(1)
			for i := 0; i < 8; i++ {
				fmt.Println(i)
				seed := rng.Uint32()
				if seed == 0 {
					seed = 1
				}
				result := txt.Text(false, h+1, seed)
				if result < 0 {
					continue
				}
				histogram[h][result-1]++
			}
			i := 8
			//optimize:
			for i < 64 {
				fmt.Println(i)
				seed := rng.Uint32()
				if seed == 0 {
					seed = 1
				}
				result := txt.Text(false, h+1, seed)
				if result < 0 {
					continue
				}
				histogram[h][result-1]++
				avg := 0.0
				for _, v := range histogram[h] {
					avg += float64(v)
				}
				avg /= float64(len(histogram[h]))
				stddev := 0.0
				for _, v := range histogram[h] {
					diff := float64(v) - avg
					stddev += diff * diff
				}
				stddev /= float64(len(histogram[h]))
				stddev = 1.7 * math.Sqrt(stddev)
				points = append(points, plotter.XY{X: float64(i), Y: stddev})
				for _, v := range histogram[h] {
					diff := float64(v) - avg
					if diff < 0 {
						continue
					}
					if diff > stddev {
						//break optimize
					}
				}
				i++
				if i%32 == 0 {
					for i := range histogram[h] {
						histogram[h][i] /= 2
					}
				}
			}

			p := plot.New()
			p.Title.Text = "iteration vs stddev"
			p.X.Label.Text = "iteration"
			p.Y.Label.Text = "stddev"

			scatter, err := plotter.NewScatter(points)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			p.Add(scatter)

			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%d_stddev.png", h))
			if err != nil {
				panic(err)
			}
		}
		for h := range histogram {
			fmt.Println(histogram[h])
		}
	} else if *FlagFinesse {
		seed := uint32(2)
		for {
			correct := 0
			for i := 1; i < 5; i++ {
				result := txt.Text2(false, i, seed)
				if result == i {
					correct++
				}
			}
			if txt.Text2(false, 5, seed) == 4 {
				correct++
			}
			fmt.Println("correct", correct)
			if correct == 5 {
				break
			}
			seed++
		}
		fmt.Println("seed", seed)
	}
}
