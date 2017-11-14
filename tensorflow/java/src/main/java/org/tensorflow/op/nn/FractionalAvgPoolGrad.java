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

public final class FractionalAvgPoolGrad<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param overlapping
     **/
    public Options overlapping(Boolean overlapping) {
      this.overlapping = overlapping;
      return this;
    }
    
    private Boolean overlapping;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new FractionalAvgPoolGrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param origInputTensorShape
   * @param outBackprop
   * @param rowPoolingSequence
   * @param colPoolingSequence
   * @return a new instance of FractionalAvgPoolGrad
   **/
  public static <T> FractionalAvgPoolGrad<T> create(Scope scope, Operand<Long> origInputTensorShape, Operand<T> outBackprop, Operand<Long> rowPoolingSequence, Operand<Long> colPoolingSequence) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FractionalAvgPoolGrad", scope.makeOpName("FractionalAvgPoolGrad"));
    opBuilder.addInput(origInputTensorShape.asOutput());
    opBuilder.addInput(outBackprop.asOutput());
    opBuilder.addInput(rowPoolingSequence.asOutput());
    opBuilder.addInput(colPoolingSequence.asOutput());
    return new FractionalAvgPoolGrad<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new FractionalAvgPoolGrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param origInputTensorShape
   * @param outBackprop
   * @param rowPoolingSequence
   * @param colPoolingSequence
   * @param options an object holding optional attributes values
   * @return a new instance of FractionalAvgPoolGrad
   **/
  public static <T> FractionalAvgPoolGrad<T> create(Scope scope, Operand<Long> origInputTensorShape, Operand<T> outBackprop, Operand<Long> rowPoolingSequence, Operand<Long> colPoolingSequence, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("FractionalAvgPoolGrad", scope.makeOpName("FractionalAvgPoolGrad"));
    opBuilder.addInput(origInputTensorShape.asOutput());
    opBuilder.addInput(outBackprop.asOutput());
    opBuilder.addInput(rowPoolingSequence.asOutput());
    opBuilder.addInput(colPoolingSequence.asOutput());
    if (options.overlapping != null) {
      opBuilder.setAttr("overlapping", options.overlapping);
    }
    return new FractionalAvgPoolGrad<T>(opBuilder.build());
  }
  
  /**
   * @param overlapping
   **/
  public static Options overlapping(Boolean overlapping) {
    return new Options().overlapping(overlapping);
  }
  
  public Output<T> output() {
    return output;
  }
  
  @Override
  public Output<T> asOutput() {
    return output;
  }
  
  private Output<T> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private FractionalAvgPoolGrad(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
