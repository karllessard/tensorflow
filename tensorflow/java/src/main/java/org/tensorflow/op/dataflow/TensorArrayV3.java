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

public final class TensorArrayV3 extends PrimitiveOp {
  
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
    
    /**
     * @param dynamicSize
     **/
    public Options dynamicSize(Boolean dynamicSize) {
      this.dynamicSize = dynamicSize;
      return this;
    }
    
    /**
     * @param clearAfterRead
     **/
    public Options clearAfterRead(Boolean clearAfterRead) {
      this.clearAfterRead = clearAfterRead;
      return this;
    }
    
    /**
     * @param tensorArrayName
     **/
    public Options tensorArrayName(String tensorArrayName) {
      this.tensorArrayName = tensorArrayName;
      return this;
    }
    
    private Shape elementShape;
    private Boolean dynamicSize;
    private Boolean clearAfterRead;
    private String tensorArrayName;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new TensorArrayV3 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param size
   * @param dtype
   * @return a new instance of TensorArrayV3
   **/
  public static <T> TensorArrayV3 create(Scope scope, Operand<Integer> size, Class<T> dtype) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArrayV3", scope.makeOpName("TensorArrayV3"));
    opBuilder.addInput(size.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    return new TensorArrayV3(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new TensorArrayV3 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param size
   * @param dtype
   * @param options an object holding optional attributes values
   * @return a new instance of TensorArrayV3
   **/
  public static <T> TensorArrayV3 create(Scope scope, Operand<Integer> size, Class<T> dtype, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("TensorArrayV3", scope.makeOpName("TensorArrayV3"));
    opBuilder.addInput(size.asOutput());
    opBuilder.setAttr("dtype", DataType.fromClass(dtype));
    if (options.elementShape != null) {
      opBuilder.setAttr("elementShape", options.elementShape);
    }
    if (options.dynamicSize != null) {
      opBuilder.setAttr("dynamicSize", options.dynamicSize);
    }
    if (options.clearAfterRead != null) {
      opBuilder.setAttr("clearAfterRead", options.clearAfterRead);
    }
    if (options.tensorArrayName != null) {
      opBuilder.setAttr("tensorArrayName", options.tensorArrayName);
    }
    return new TensorArrayV3(opBuilder.build());
  }
  
  /**
   * @param elementShape
   **/
  public static Options elementShape(Shape elementShape) {
    return new Options().elementShape(elementShape);
  }
  
  /**
   * @param dynamicSize
   **/
  public static Options dynamicSize(Boolean dynamicSize) {
    return new Options().dynamicSize(dynamicSize);
  }
  
  /**
   * @param clearAfterRead
   **/
  public static Options clearAfterRead(Boolean clearAfterRead) {
    return new Options().clearAfterRead(clearAfterRead);
  }
  
  /**
   * @param tensorArrayName
   **/
  public static Options tensorArrayName(String tensorArrayName) {
    return new Options().tensorArrayName(tensorArrayName);
  }
  
  public Output<?> handle() {
    return handle;
  }
  
  public Output<Float> flow() {
    return flow;
  }
  
  private Output<?> handle;
  private Output<Float> flow;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private TensorArrayV3(Operation operation) {
    super(operation);
    int outputIdx = 0;
    handle = operation.output(outputIdx++);
    flow = operation.output(outputIdx++);
  }
}
