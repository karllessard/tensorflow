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
package org.tensorflow.op.training;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class ResourceSparseApplyProximalGradientDescent extends PrimitiveOp {
  
  /**
   * Class holding optional attributes of this operation
   **/
  public static class Options {
    
    /**
     * @param useLocking
     **/
    public Options useLocking(Boolean useLocking) {
      this.useLocking = useLocking;
      return this;
    }
    
    private Boolean useLocking;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ResourceSparseApplyProximalGradientDescent operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param alpha
   * @param l1
   * @param l2
   * @param grad
   * @param indices
   * @return a new instance of ResourceSparseApplyProximalGradientDescent
   **/
  public static <T, U> ResourceSparseApplyProximalGradientDescent create(Scope scope, Operand<?> var, Operand<T> alpha, Operand<T> l1, Operand<T> l2, Operand<T> grad, Operand<U> indices) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceSparseApplyProximalGradientDescent", scope.makeOpName("ResourceSparseApplyProximalGradientDescent"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(alpha.asOutput());
    opBuilder.addInput(l1.asOutput());
    opBuilder.addInput(l2.asOutput());
    opBuilder.addInput(grad.asOutput());
    opBuilder.addInput(indices.asOutput());
    return new ResourceSparseApplyProximalGradientDescent(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ResourceSparseApplyProximalGradientDescent operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param alpha
   * @param l1
   * @param l2
   * @param grad
   * @param indices
   * @param options an object holding optional attributes values
   * @return a new instance of ResourceSparseApplyProximalGradientDescent
   **/
  public static <T, U> ResourceSparseApplyProximalGradientDescent create(Scope scope, Operand<?> var, Operand<T> alpha, Operand<T> l1, Operand<T> l2, Operand<T> grad, Operand<U> indices, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceSparseApplyProximalGradientDescent", scope.makeOpName("ResourceSparseApplyProximalGradientDescent"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(alpha.asOutput());
    opBuilder.addInput(l1.asOutput());
    opBuilder.addInput(l2.asOutput());
    opBuilder.addInput(grad.asOutput());
    opBuilder.addInput(indices.asOutput());
    if (options.useLocking != null) {
      opBuilder.setAttr("useLocking", options.useLocking);
    }
    return new ResourceSparseApplyProximalGradientDescent(opBuilder.build());
  }
  
  /**
   * @param useLocking
   **/
  public static Options useLocking(Boolean useLocking) {
    return new Options().useLocking(useLocking);
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ResourceSparseApplyProximalGradientDescent(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
