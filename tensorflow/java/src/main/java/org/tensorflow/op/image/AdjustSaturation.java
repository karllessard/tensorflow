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
package org.tensorflow.op.image;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class AdjustSaturation extends PrimitiveOp implements Operand<Float> {
  
  /**
   * Factory method to create a class to wrap a new AdjustSaturation operation to the graph.
   * 
   * @param scope Current graph scope
   * @param images
   * @param scale
   * @return a new instance of AdjustSaturation
   **/
  public static AdjustSaturation create(Scope scope, Operand<Float> images, Operand<Float> scale) {
    OperationBuilder opBuilder = scope.graph().opBuilder("AdjustSaturation", scope.makeOpName("AdjustSaturation"));
    opBuilder.addInput(images.asOutput());
    opBuilder.addInput(scale.asOutput());
    return new AdjustSaturation(opBuilder.build());
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
  private AdjustSaturation(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
