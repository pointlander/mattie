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
		result := txt.Text(false, i, 76)
		if i != result {
			t.Errorf("fail %d!=%d", i, result)
		}
	}
}

func TestText2(t *testing.T) {
	result := txt.Text(false, 5, 76)
	if result != 4 {
		t.Fatalf("fail 4!=%d", result)
	}
}
