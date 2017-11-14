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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class CropAndResizeGradImage<T> extends PrimitiveOp implements Operand<T> {
  
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
   * Factory method to create a class to wrap a new CropAndResizeGradImage operation to the graph.
   * 
   * @param scope Current graph scope
   * @param grads
   * @param boxes
   * @param boxInd
   * @param imageSize
   * @param T
   * @return a new instance of CropAndResizeGradImage
   **/
  public static <T> CropAndResizeGradImage<T> create(Scope scope, Operand<Float> grads, Operand<Float> boxes, Operand<Integer> boxInd, Operand<Integer> imageSize, Class<T> T) {
    OperationBuilder opBuilder = scope.graph().opBuilder("CropAndResizeGradImage", scope.makeOpName("CropAndResizeGradImage"));
    opBuilder.addInput(grads.asOutput());
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(boxInd.asOutput());
    opBuilder.addInput(imageSize.asOutput());
    opBuilder.setAttr("T", DataType.fromClass(T));
    return new CropAndResizeGradImage<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new CropAndResizeGradImage operation to the graph.
   * 
   * @param scope Current graph scope
   * @param grads
   * @param boxes
   * @param boxInd
   * @param imageSize
   * @param T
   * @param options an object holding optional attributes values
   * @return a new instance of CropAndResizeGradImage
   **/
  public static <T> CropAndResizeGradImage<T> create(Scope scope, Operand<Float> grads, Operand<Float> boxes, Operand<Integer> boxInd, Operand<Integer> imageSize, Class<T> T, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("CropAndResizeGradImage", scope.makeOpName("CropAndResizeGradImage"));
    opBuilder.addInput(grads.asOutput());
    opBuilder.addInput(boxes.asOutput());
    opBuilder.addInput(boxInd.asOutput());
    opBuilder.addInput(imageSize.asOutput());
    opBuilder.setAttr("T", DataType.fromClass(T));
    if (options.method != null) {
      opBuilder.setAttr("method", options.method);
    }
    return new CropAndResizeGradImage<T>(opBuilder.build());
  }
  
  /**
   * @param method
   **/
  public static Options method(String method) {
    return new Options().method(method);
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
  private CropAndResizeGradImage(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
