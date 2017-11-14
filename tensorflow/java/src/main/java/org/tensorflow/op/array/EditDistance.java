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

public final class EditDistance extends PrimitiveOp implements Operand<Float> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param normalize
     **/
    public Options normalize(Boolean normalize) {
      this.normalize = normalize;
      return this;
    }
    
    private Boolean normalize;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new EditDistance operation to the graph.
   * 
   * @param scope Current graph scope
   * @param hypothesisIndices
   * @param hypothesisValues
   * @param hypothesisShape
   * @param truthIndices
   * @param truthValues
   * @param truthShape
   * @return a new instance of EditDistance
   **/
  public static <T> EditDistance create(Scope scope, Operand<Long> hypothesisIndices, Operand<T> hypothesisValues, Operand<Long> hypothesisShape, Operand<Long> truthIndices, Operand<T> truthValues, Operand<Long> truthShape) {
    OperationBuilder opBuilder = scope.graph().opBuilder("EditDistance", scope.makeOpName("EditDistance"));
    opBuilder.addInput(hypothesisIndices.asOutput());
    opBuilder.addInput(hypothesisValues.asOutput());
    opBuilder.addInput(hypothesisShape.asOutput());
    opBuilder.addInput(truthIndices.asOutput());
    opBuilder.addInput(truthValues.asOutput());
    opBuilder.addInput(truthShape.asOutput());
    return new EditDistance(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new EditDistance operation to the graph.
   * 
   * @param scope Current graph scope
   * @param hypothesisIndices
   * @param hypothesisValues
   * @param hypothesisShape
   * @param truthIndices
   * @param truthValues
   * @param truthShape
   * @param options an object holding optional attributes values
   * @return a new instance of EditDistance
   **/
  public static <T> EditDistance create(Scope scope, Operand<Long> hypothesisIndices, Operand<T> hypothesisValues, Operand<Long> hypothesisShape, Operand<Long> truthIndices, Operand<T> truthValues, Operand<Long> truthShape, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("EditDistance", scope.makeOpName("EditDistance"));
    opBuilder.addInput(hypothesisIndices.asOutput());
    opBuilder.addInput(hypothesisValues.asOutput());
    opBuilder.addInput(hypothesisShape.asOutput());
    opBuilder.addInput(truthIndices.asOutput());
    opBuilder.addInput(truthValues.asOutput());
    opBuilder.addInput(truthShape.asOutput());
    if (options.normalize != null) {
      opBuilder.setAttr("normalize", options.normalize);
    }
    return new EditDistance(opBuilder.build());
  }
  
  /**
   * @param normalize
   **/
  public static Options normalize(Boolean normalize) {
    return new Options().normalize(normalize);
  }
  
  public Output<Float> output() {
    return output;
  }
  
  @Override
  public Output<Float> asOutput() {
    return output;
  }
  
  private Output<Float> output;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private EditDistance(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
