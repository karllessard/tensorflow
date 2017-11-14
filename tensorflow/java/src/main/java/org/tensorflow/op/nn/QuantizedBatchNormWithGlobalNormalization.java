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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QuantizedBatchNormWithGlobalNormalization<U> extends PrimitiveOp {
  
  /**
   * Factory method to create a class to wrap a new QuantizedBatchNormWithGlobalNormalization operation to the graph.
   * 
   * @param scope Current graph scope
   * @param t
   * @param tMin
   * @param tMax
   * @param m
   * @param mMin
   * @param mMax
   * @param v
   * @param vMin
   * @param vMax
   * @param beta
   * @param betaMin
   * @param betaMax
   * @param gamma
   * @param gammaMin
   * @param gammaMax
   * @param out_type
   * @param varianceEpsilon
   * @param scaleAfterNormalization
   * @return a new instance of QuantizedBatchNormWithGlobalNormalization
   **/
  public static <T, U> QuantizedBatchNormWithGlobalNormalization<U> create(Scope scope, Operand<T> t, Operand<Float> tMin, Operand<Float> tMax, Operand<T> m, Operand<Float> mMin, Operand<Float> mMax, Operand<T> v, Operand<Float> vMin, Operand<Float> vMax, Operand<T> beta, Operand<Float> betaMin, Operand<Float> betaMax, Operand<T> gamma, Operand<Float> gammaMin, Operand<Float> gammaMax, Class<U> out_type, Float varianceEpsilon, Boolean scaleAfterNormalization) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedBatchNormWithGlobalNormalization", scope.makeOpName("QuantizedBatchNormWithGlobalNormalization"));
    opBuilder.addInput(t.asOutput());
    opBuilder.addInput(tMin.asOutput());
    opBuilder.addInput(tMax.asOutput());
    opBuilder.addInput(m.asOutput());
    opBuilder.addInput(mMin.asOutput());
    opBuilder.addInput(mMax.asOutput());
    opBuilder.addInput(v.asOutput());
    opBuilder.addInput(vMin.asOutput());
    opBuilder.addInput(vMax.asOutput());
    opBuilder.addInput(beta.asOutput());
    opBuilder.addInput(betaMin.asOutput());
    opBuilder.addInput(betaMax.asOutput());
    opBuilder.addInput(gamma.asOutput());
    opBuilder.addInput(gammaMin.asOutput());
    opBuilder.addInput(gammaMax.asOutput());
    opBuilder.setAttr("out_type", DataType.fromClass(out_type));
    opBuilder.setAttr("varianceEpsilon", varianceEpsilon);
    opBuilder.setAttr("scaleAfterNormalization", scaleAfterNormalization);
    return new QuantizedBatchNormWithGlobalNormalization<U>(opBuilder.build());
  }
  
  public Output<U> result() {
    return result;
  }
  
  public Output<Float> resultMin() {
    return resultMin;
  }
  
  public Output<Float> resultMax() {
    return resultMax;
  }
  
  private Output<U> result;
  private Output<Float> resultMin;
  private Output<Float> resultMax;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedBatchNormWithGlobalNormalization(Operation operation) {
    super(operation);
    int outputIdx = 0;
    result = operation.output(outputIdx++);
    resultMin = operation.output(outputIdx++);
    resultMax = operation.output(outputIdx++);
  }
}
