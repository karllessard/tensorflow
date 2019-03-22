/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow;

import static org.junit.Assert.assertEquals;

import java.nio.IntBuffer;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.linalg.MatMul;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Div;
import org.tensorflow.op.math.Pow;

@RunWith(JUnit4.class)
public class EagerSessionTest {
 
  @Test
  public void scalarValue() {
    try (EagerSession s = EagerSession.create()) {
      Ops ops = Ops.create(s);

      Constant<Long> a = ops.constant(4L);
      assertEquals(4L, a.asOutput().tensor().longValue());
    }
  }

  @Test
  public void scalarComputation() {
    try (EagerSession session = EagerSession.create()) {
      Ops ops = Ops.create(session);

      Constant<Long> a = ops.constant(4L);
      Pow<Long> pow = ops.math.pow(a, a);
      assertEquals(256L, pow.z().tensor().longValue());
    }
  }

  @Test
  public void arrayComputation() {
    try (EagerSession session = EagerSession.create()) {
      Ops ops = Ops.create(session);

      Constant<Integer> a = ops.constant(new int[]{1, 2});
      Add<Integer> add = ops.math.add(a, ops.constant(2));
      assertEquals(IntBuffer.wrap(new int[] {3, 4}), add.z().tensor().buffer().asIntBuffer());
    }
  }

  @Test
  public void matrixComputation() {
    try (EagerSession session = EagerSession.create()) {
      Ops ops = Ops.create(session);

      Constant<Integer> a = ops.constant(new int[][]{ {1, 2}, {3, 4} });
      MatMul<Integer> matmul = ops.linalg.matMul(a, a, MatMul.transposeB(true));
      assertEquals(IntBuffer.wrap(new int[] {5, 11, 11, 25}), matmul.product().tensor().buffer().asIntBuffer());
    }
  }

  @Test
  public void chainedComputation() {
    try (EagerSession session = EagerSession.create()) {
      Ops ops = Ops.create(session);

      Constant<Float> a = ops.constant(4.0f);
      Constant<Float> b = ops.constant(2.0f);
      Pow<Float> pow = ops.math.pow(a, a);
      Div<Float> div = ops.math.div(pow, b);
      assertEquals(128.0f, div.z().tensor().floatValue(), 0.0f);
    }
  }
}
