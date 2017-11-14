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

public final class TensorArrayConcatV2<T> extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param elementShapeExcept0
     **/
    public Options elementShapeExcept0(Shape elementShapeExcept0) {
      this.elementShapeExcept0 = elementShapeExcept0;
      return this;
    }
    
    private Shape elementShapeExcept0;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new TensorArrayConcatV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param flowIn
   * @param dtype
   * @return a new instance of TensorArrayConcatV2
   **/
  public static <T> TensorArrayConcatV2<T> create(Scope scope, Operand<String> handle, Operand<Float> flowIn, Class<T> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArrayConcatV2", scope.makeOpName("TensorArrayConcatV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(flowIn.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new TensorArrayConcatV2<T>(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new TensorArrayConcatV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param handle
   * @param flowIn
   * @param dtype
   * @param options an object holding optional attributes values
   * @return a new instance of TensorArrayConcatV2
   **/
  public static <T> TensorArrayConcatV2<T> create(Scope scope, Operand<String> handle, Operand<Float> flowIn, Class<T> dtype, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArrayConcatV2", scope.makeOpName("TensorArrayConcatV2"));
    opBuilder.addInput(handle.asOutput());
    opBuilder.addInput(flowIn.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    if (options.elementShapeExcept0 != null) {
      opBuilder.setAttr("elementShapeExcept0", options.elementShapeExcept0);
    }
    return new TensorArrayConcatV2<T>(opBuilder.build());
  }
  
  /**
   * @param elementShapeExcept0
   **/
  public static Options elementShapeExcept0(Shape elementShapeExcept0) {
    return new Options().elementShapeExcept0(elementShapeExcept0);
  }
  
  public Output<T> value() {
    return value;
  }
  
  public Output<Long> lengths() {
    return lengths;
  }
  
  private Output<T> value;
  private Output<Long> lengths;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TensorArrayConcatV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    value = operation.output(outputIdx++);
    lengths = operation.output(outputIdx++);
  }
}
