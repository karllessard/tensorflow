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

public final class ResourceApplyProximalAdagrad extends PrimitiveOp {
  
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
   * Factory method to create a class to wrap a new ResourceApplyProximalAdagrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param accum
   * @param lr
   * @param l1
   * @param l2
   * @param grad
   * @return a new instance of ResourceApplyProximalAdagrad
   **/
  public static <T> ResourceApplyProximalAdagrad create(Scope scope, Operand<?> var, Operand<?> accum, Operand<T> lr, Operand<T> l1, Operand<T> l2, Operand<T> grad) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceApplyProximalAdagrad", scope.makeOpName("ResourceApplyProximalAdagrad"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(accum.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(l1.asOutput());
    opBuilder.addInput(l2.asOutput());
    opBuilder.addInput(grad.asOutput());
    return new ResourceApplyProximalAdagrad(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ResourceApplyProximalAdagrad operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param accum
   * @param lr
   * @param l1
   * @param l2
   * @param grad
   * @param options an object holding optional attributes values
   * @return a new instance of ResourceApplyProximalAdagrad
   **/
  public static <T> ResourceApplyProximalAdagrad create(Scope scope, Operand<?> var, Operand<?> accum, Operand<T> lr, Operand<T> l1, Operand<T> l2, Operand<T> grad, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceApplyProximalAdagrad", scope.makeOpName("ResourceApplyProximalAdagrad"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(accum.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(l1.asOutput());
    opBuilder.addInput(l2.asOutput());
    opBuilder.addInput(grad.asOutput());
    if (options.useLocking != null) {
      opBuilder.setAttr("useLocking", options.useLocking);
    }
    return new ResourceApplyProximalAdagrad(opBuilder.build());
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
  private ResourceApplyProximalAdagrad(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
