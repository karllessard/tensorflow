/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
package org.tensorflow.op.nn;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class BatchNormWithGlobalNormalizationGrad<T> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new BatchNormWithGlobalNormalizationGrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param t
   * @param m
   * @param v
   * @param gamma
   * @param backprop
   * @param varianceEpsilon
   * @param scaleAfterNormalization
   * @return a new instance of BatchNormWithGlobalNormalizationGrad
   **/
  public static <T> BatchNormWithGlobalNormalizationGrad<T> create(Scope scope, Operand<T> t, Operand<T> m, Operand<T> v, Operand<T> gamma, Operand<T> backprop, Float varianceEpsilon, Boolean scaleAfterNormalization) {
    OperationBuilder opBuilder = scope.graph().opBuilder("BatchNormWithGlobalNormalizationGrad", scope.makeOpName("BatchNormWithGlobalNormalizationGrad"));
    opBuilder.addInput(t.asOutput());
    opBuilder.addInput(m.asOutput());
    opBuilder.addInput(v.asOutput());
    opBuilder.addInput(gamma.asOutput());
    opBuilder.addInput(backprop.asOutput());
    opBuilder.setAttr("varianceEpsilon", varianceEpsilon);
    opBuilder.setAttr("scaleAfterNormalization", scaleAfterNormalization);
    return new BatchNormWithGlobalNormalizationGrad<T>(opBuilder.build());
  }
  
  public Output<T> dx() {
    return dx;
  }
  
  public Output<T> dm() {
    return dm;
  }
  
  public Output<T> dv() {
    return dv;
  }
  
  public Output<T> db() {
    return db;
  }
  
  public Output<T> dg() {
    return dg;
  }
  
  private Output<T> dx;
  private Output<T> dm;
  private Output<T> dv;
  private Output<T> db;
  private Output<T> dg;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private BatchNormWithGlobalNormalizationGrad(Operation operation) {
    super(operation);
    int outputIdx = 0;
    dx = operation.output(outputIdx++);
    dm = operation.output(outputIdx++);
    dv = operation.output(outputIdx++);
    db = operation.output(outputIdx++);
    dg = operation.output(outputIdx++);
  }
}
