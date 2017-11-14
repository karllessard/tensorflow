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

public final class ResourceApplyAdam extends PrimitiveOp {
  
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
    
    /**
     * @param useNesterov
     **/
    public Options useNesterov(Boolean useNesterov) {
      this.useNesterov = useNesterov;
      return this;
    }
    
    private Boolean useLocking;
    private Boolean useNesterov;
    
    private Options() {
    }
  }
  
  /**
   * Factory method to create a class to wrap a new ResourceApplyAdam operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param m
   * @param v
   * @param beta1Power
   * @param beta2Power
   * @param lr
   * @param beta1
   * @param beta2
   * @param epsilon
   * @param grad
   * @return a new instance of ResourceApplyAdam
   **/
  public static <T> ResourceApplyAdam create(Scope scope, Operand<?> var, Operand<?> m, Operand<?> v, Operand<T> beta1Power, Operand<T> beta2Power, Operand<T> lr, Operand<T> beta1, Operand<T> beta2, Operand<T> epsilon, Operand<T> grad) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceApplyAdam", scope.makeOpName("ResourceApplyAdam"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(m.asOutput());
    opBuilder.addInput(v.asOutput());
    opBuilder.addInput(beta1Power.asOutput());
    opBuilder.addInput(beta2Power.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(beta1.asOutput());
    opBuilder.addInput(beta2.asOutput());
    opBuilder.addInput(epsilon.asOutput());
    opBuilder.addInput(grad.asOutput());
    return new ResourceApplyAdam(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ResourceApplyAdam operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param m
   * @param v
   * @param beta1Power
   * @param beta2Power
   * @param lr
   * @param beta1
   * @param beta2
   * @param epsilon
   * @param grad
   * @param options an object holding optional attributes values
   * @return a new instance of ResourceApplyAdam
   **/
  public static <T> ResourceApplyAdam create(Scope scope, Operand<?> var, Operand<?> m, Operand<?> v, Operand<T> beta1Power, Operand<T> beta2Power, Operand<T> lr, Operand<T> beta1, Operand<T> beta2, Operand<T> epsilon, Operand<T> grad, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceApplyAdam", scope.makeOpName("ResourceApplyAdam"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(m.asOutput());
    opBuilder.addInput(v.asOutput());
    opBuilder.addInput(beta1Power.asOutput());
    opBuilder.addInput(beta2Power.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(beta1.asOutput());
    opBuilder.addInput(beta2.asOutput());
    opBuilder.addInput(epsilon.asOutput());
    opBuilder.addInput(grad.asOutput());
    if (options.useLocking != null) {
      opBuilder.setAttr("useLocking", options.useLocking);
    }
    if (options.useNesterov != null) {
      opBuilder.setAttr("useNesterov", options.useNesterov);
    }
    return new ResourceApplyAdam(opBuilder.build());
  }
  
  /**
   * @param useLocking
   **/
  public static Options useLocking(Boolean useLocking) {
    return new Options().useLocking(useLocking);
  }
  
  /**
   * @param useNesterov
   **/
  public static Options useNesterov(Boolean useNesterov) {
    return new Options().useNesterov(useNesterov);
  }
  
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private ResourceApplyAdam(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
