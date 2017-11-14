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
package org.tensorflow.op.state;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ResourceCountUpTo<T> extends PrimitiveOp implements Operand<T> {
  
  /**
   * Factory method to create a class to wrap a new ResourceCountUpTo operation to the graph.
   * 
   * @param scope Current graph scope
   * @param resource
   * @param limit
   * @param T
   * @return a new instance of ResourceCountUpTo
   **/
  public static <T> ResourceCountUpTo<T> create(Scope scope, Operand<?> resource, Integer limit, Class<T> T) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceCountUpTo", scope.makeOpName("ResourceCountUpTo"));
    opBuilder.addInput(resource.asOutput());
    opBuilder.setAttr("limit", limit);
    opBuilder.setAttr("T", DataType.fromClass(T));
    return new ResourceCountUpTo<T>(opBuilder.build());
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
  private ResourceCountUpTo(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
