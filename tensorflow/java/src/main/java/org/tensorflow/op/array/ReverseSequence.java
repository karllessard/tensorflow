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
package org.tensorflow.op.array;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ReverseSequence<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param batchDim
     **/
    public Options batchDim(Integer batchDim) {
      this.batchDim = batchDim;
      return this;
    }
    
    private Integer batchDim;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ReverseSequence operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param seqLengths
   * @param seqDim
   * @return a new instance of ReverseSequence
   **/
  public static <T, U> ReverseSequence<T> create(Scope scope, Operand<T> input, Operand<U> seqLengths, Integer seqDim) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReverseSequence", scope.makeOpName("ReverseSequence"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(seqLengths.asOutput());
    opBuilder.setAttr("seqDim", seqDim);
    return new ReverseSequence<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ReverseSequence operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param seqLengths
   * @param seqDim
   * @param options an object holding optional attributes values
   * @return a new instance of ReverseSequence
   **/
  public static <T, U> ReverseSequence<T> create(Scope scope, Operand<T> input, Operand<U> seqLengths, Integer seqDim, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ReverseSequence", scope.makeOpName("ReverseSequence"));
    opBuilder.addInput(input.asOutput());
    opBuilder.addInput(seqLengths.asOutput());
    opBuilder.setAttr("seqDim", seqDim);
    if (options.batchDim != null) {
      opBuilder.setAttr("batchDim", options.batchDim);
    }
    return new ReverseSequence<T>(opBuilder.build());
  }
  
  /**
   * @param batchDim
   **/
  public static Options batchDim(Integer batchDim) {
    return new Options().batchDim(batchDim);
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
  private ReverseSequence(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
