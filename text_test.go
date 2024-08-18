// Copyright 2024 The Mattie Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/pointlander/mattie/txt"
)

func TestText(t *testing.T) {
	for i := 1; i < 5; i++ {
		result := txt.Text(false, i)
		if i != result {
			t.Fatalf("fail %d!=%d", i, result)
		}
	}
}
