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

import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class Squeeze<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param squeezeDims
     **/
    public Options squeezeDims(List<Integer> squeezeDims) {
      this.squeezeDims = squeezeDims;
      return this;
    }
    
    private List<Integer> squeezeDims;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new Squeeze operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @return a new instance of Squeeze
   **/
  public static <T> Squeeze<T> create(Scope scope, Operand<T> input) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Squeeze", scope.makeOpName("Squeeze"));
    opBuilder.addInput(input.asOutput());
    return new Squeeze<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new Squeeze operation to the graph.
   * 
   * @param scope Current graph scope
   * @param input
   * @param options an object holding optional attributes values
   * @return a new instance of Squeeze
   **/
  public static <T> Squeeze<T> create(Scope scope, Operand<T> input, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("Squeeze", scope.makeOpName("Squeeze"));
    opBuilder.addInput(input.asOutput());
    if (options.squeezeDims != null) {
      long[] squeezeDimsArray = new long[options.squeezeDims.size()];
      for (int i = 0; i < squeezeDimsArray.length; ++i) {
        squeezeDimsArray[i] = options.squeezeDims.get(i);
      }
      opBuilder.setAttr("squeezeDims", squeezeDimsArray);
    }
    return new Squeeze<T>(opBuilder.build());
  }
  
  /**
   * @param squeezeDims
   **/
  public static Options squeezeDims(List<Integer> squeezeDims) {
    return new Options().squeezeDims(squeezeDims);
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
  private Squeeze(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
