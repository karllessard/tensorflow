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

public final class CropAndResizeGradBoxes extends PrimitiveOp implements Operand<Float> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param method
     **/
    public Options method(String method) {
      this.method = method;
      return this;
    }
    
    private String method;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new CropAndResizeGradBoxes operation to the graph.
   * 
   * @param scope Current graph scope
   * @param grads
   * @param image
   * @param boxes
   * @param boxInd
   * @return a new instance of CropAndResizeGradBoxes
   **/
  public static <T> CropAndResizeGradBoxes create(Scope scope, Operand<Float> grads, Operand<T> image, Operand<Float> boxes, Operand<Integer> boxInd) {
    OperationBuilder opBuilder = scope.graph().opBuilder("CropAndResizeGradBoxes", scope.makeOpName("CropAndResizeGradBoxes"));
    opBuilder.addInput(grads.asOutput());
    opBuilder.addInput(image.asOutput());
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(boxInd.asOutput());
    return new CropAndResizeGradBoxes(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new CropAndResizeGradBoxes operation to the graph.
   * 
   * @param scope Current graph scope
   * @param grads
   * @param image
   * @param boxes
   * @param boxInd
   * @param options an object holding optional attributes values
   * @return a new instance of CropAndResizeGradBoxes
   **/
  public static <T> CropAndResizeGradBoxes create(Scope scope, Operand<Float> grads, Operand<T> image, Operand<Float> boxes, Operand<Integer> boxInd, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("CropAndResizeGradBoxes", scope.makeOpName("CropAndResizeGradBoxes"));
    opBuilder.addInput(grads.asOutput());
    opBuilder.addInput(image.asOutput());
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(boxInd.asOutput());
    if (options.method != null) {
      opBuilder.setAttr("method", options.method);
    }
    return new CropAndResizeGradBoxes(opBuilder.build());
  }
  
  /**
   * @param method
   **/
  public static Options method(String method) {
    return new Options().method(method);
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
  private CropAndResizeGradBoxes(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
