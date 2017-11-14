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
package org.tensorflow.op.dataflow;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.Shape;
import org.tensorflow.op.Scope;

public final class TensorArrayGatherV2<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param elementShape
     **/
    public Options elementShape(Shape elementShape) {
      this.elementShape = elementShape;
      return this;
    }
    
    private Shape elementShape;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new TensorArrayGatherV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param indices
   * @param flowIn
   * @param dtype
   * @return a new instance of TensorArrayGatherV2
   **/
  public static <T> TensorArrayGatherV2<T> create(Scope scope, Operand<String> handle, Operand<Integer> indices, Operand<Float> flowIn, Class<T> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArrayGatherV2", scope.makeOpName("TensorArrayGatherV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(indices.asOutput());
    opBuilder.addInput(flowIn.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new TensorArrayGatherV2<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new TensorArrayGatherV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param indices
   * @param flowIn
   * @param dtype
   * @param options an object holding optional attributes values
   * @return a new instance of TensorArrayGatherV2
   **/
  public static <T> TensorArrayGatherV2<T> create(Scope scope, Operand<String> handle, Operand<Integer> indices, Operand<Float> flowIn, Class<T> dtype, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArrayGatherV2", scope.makeOpName("TensorArrayGatherV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(indices.asOutput());
    opBuilder.addInput(flowIn.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    if (options.elementShape != null) {
      opBuilder.setAttr("elementShape", options.elementShape);
    }
    return new TensorArrayGatherV2<T>(opBuilder.build());
  }
  
  /**
   * @param elementShape
   **/
  public static Options elementShape(Shape elementShape) {
    return new Options().elementShape(elementShape);
  }
  
  public Output<T> value() {
    return value;
  }
  
  @Override
  public Output<T> asOutput() {
    return value;
  }
  
  private Output<T> value;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TensorArrayGatherV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    value = operation.output(outputIdx++);
  }
}
