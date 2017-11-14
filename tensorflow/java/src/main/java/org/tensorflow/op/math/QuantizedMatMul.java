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
package org.tensorflow.op.math;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class QuantizedMatMul<V> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param transposeA
     **/
    public Options transposeA(Boolean transposeA) {
      this.transposeA = transposeA;
      return this;
    }
    
    /**
     * @param transposeB
     **/
    public Options transposeB(Boolean transposeB) {
      this.transposeB = transposeB;
      return this;
    }
    
    private Boolean transposeA;
    private Boolean transposeB;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizedMatMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param a
   * @param b
   * @param minA
   * @param maxA
   * @param minB
   * @param maxB
   * @param Toutput
   * @param Tactivation
   * @return a new instance of QuantizedMatMul
   **/
  public static <T, U, V, W> QuantizedMatMul<V> create(Scope scope, Operand<T> a, Operand<U> b, Operand<Float> minA, Operand<Float> maxA, Operand<Float> minB, Operand<Float> maxB, Class<V> Toutput, Class<W> Tactivation) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedMatMul", scope.makeOpName("QuantizedMatMul"));
    opBuilder.addInput(a.asOutput());
    opBuilder.addInput(b.asOutput());
    opBuilder.addInput(minA.asOutput());
    opBuilder.addInput(maxA.asOutput());
    opBuilder.addInput(minB.asOutput());
    opBuilder.addInput(maxB.asOutput());
    opBuilder.setAttr("Toutput", DataType.fromClass(Toutput));
    opBuilder.setAttr("Tactivation", DataType.fromClass(Tactivation));
    return new QuantizedMatMul<V>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new QuantizedMatMul operation to the graph.
   * 
   * @param scope Current graph scope
   * @param a
   * @param b
   * @param minA
   * @param maxA
   * @param minB
   * @param maxB
   * @param Toutput
   * @param Tactivation
   * @param options an object holding optional attributes values
   * @return a new instance of QuantizedMatMul
   **/
  public static <T, U, V, W> QuantizedMatMul<V> create(Scope scope, Operand<T> a, Operand<U> b, Operand<Float> minA, Operand<Float> maxA, Operand<Float> minB, Operand<Float> maxB, Class<V> Toutput, Class<W> Tactivation, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("QuantizedMatMul", scope.makeOpName("QuantizedMatMul"));
    opBuilder.addInput(a.asOutput());
    opBuilder.addInput(b.asOutput());
    opBuilder.addInput(minA.asOutput());
    opBuilder.addInput(maxA.asOutput());
    opBuilder.addInput(minB.asOutput());
    opBuilder.addInput(maxB.asOutput());
    opBuilder.setAttr("Toutput", DataType.fromClass(Toutput));
    opBuilder.setAttr("Tactivation", DataType.fromClass(Tactivation));
    if (options.transposeA != null) {
      opBuilder.setAttr("transposeA", options.transposeA);
    }
    if (options.transposeB != null) {
      opBuilder.setAttr("transposeB", options.transposeB);
    }
    return new QuantizedMatMul<V>(opBuilder.build());
  }
  
  /**
   * @param transposeA
   **/
  public static Options transposeA(Boolean transposeA) {
    return new Options().transposeA(transposeA);
  }
  
  /**
   * @param transposeB
   **/
  public static Options transposeB(Boolean transposeB) {
    return new Options().transposeB(transposeB);
  }
  
  public Output<V> out() {
    return out;
  }
  
  public Output<Float> minOut() {
    return minOut;
  }
  
  public Output<Float> maxOut() {
    return maxOut;
  }
  
  private Output<V> out;
  private Output<Float> minOut;
  private Output<Float> maxOut;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private QuantizedMatMul(Operation operation) {
    super(operation);
    int outputIdx = 0;
    out = operation.output(outputIdx++);
    minOut = operation.output(outputIdx++);
    maxOut = operation.output(outputIdx++);
  }
}
