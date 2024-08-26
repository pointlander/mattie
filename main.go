// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"

	"github.com/pointlander/matrix"
	"github.com/pointlander/mattie/original"
	"github.com/pointlander/mattie/random"
	"github.com/pointlander/mattie/text"
	"github.com/pointlander/mattie/text2"
	"github.com/pointlander/mattie/txt"
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
			rng := matrix.Rand(1)
			for i := 0; i < 8; i++ {
				fmt.Println(i)
				seed := rng.Uint32()
				if seed == 0 {
					seed = 1
				}
				result := txt.Text(false, h+1, seed)
				histogram[h][result-1]++
			}
		}
		for h := range histogram {
			fmt.Println(histogram[h])
		}
	}
}
