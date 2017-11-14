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

public final class ResourceApplyCenteredRMSProp extends PrimitiveOp {
  
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
   * Factory method to create a class to wrap a new ResourceApplyCenteredRMSProp operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param mg
   * @param ms
   * @param mom
   * @param lr
   * @param rho
   * @param momentum
   * @param epsilon
   * @param grad
   * @return a new instance of ResourceApplyCenteredRMSProp
   **/
  public static <T> ResourceApplyCenteredRMSProp create(Scope scope, Operand<?> var, Operand<?> mg, Operand<?> ms, Operand<?> mom, Operand<T> lr, Operand<T> rho, Operand<T> momentum, Operand<T> epsilon, Operand<T> grad) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceApplyCenteredRMSProp", scope.makeOpName("ResourceApplyCenteredRMSProp"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(mg.asOutput());
    opBuilder.addInput(ms.asOutput());
    opBuilder.addInput(mom.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(rho.asOutput());
    opBuilder.addInput(momentum.asOutput());
    opBuilder.addInput(epsilon.asOutput());
    opBuilder.addInput(grad.asOutput());
    return new ResourceApplyCenteredRMSProp(opBuilder.build());
  }
  
  /**
   * Factory method to create a class to wrap a new ResourceApplyCenteredRMSProp operation to the graph.
   * 
   * @param scope Current graph scope
   * @param var
   * @param mg
   * @param ms
   * @param mom
   * @param lr
   * @param rho
   * @param momentum
   * @param epsilon
   * @param grad
   * @param options an object holding optional attributes values
   * @return a new instance of ResourceApplyCenteredRMSProp
   **/
  public static <T> ResourceApplyCenteredRMSProp create(Scope scope, Operand<?> var, Operand<?> mg, Operand<?> ms, Operand<?> mom, Operand<T> lr, Operand<T> rho, Operand<T> momentum, Operand<T> epsilon, Operand<T> grad, Options options) {
    OperationBuilder opBuilder = scope.graph().opBuilder("ResourceApplyCenteredRMSProp", scope.makeOpName("ResourceApplyCenteredRMSProp"));
    opBuilder.addInput(var.asOutput());
    opBuilder.addInput(mg.asOutput());
    opBuilder.addInput(ms.asOutput());
    opBuilder.addInput(mom.asOutput());
    opBuilder.addInput(lr.asOutput());
    opBuilder.addInput(rho.asOutput());
    opBuilder.addInput(momentum.asOutput());
    opBuilder.addInput(epsilon.asOutput());
    opBuilder.addInput(grad.asOutput());
    if (options.useLocking != null) {
      opBuilder.setAttr("useLocking", options.useLocking);
    }
    return new ResourceApplyCenteredRMSProp(opBuilder.build());
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
  private ResourceApplyCenteredRMSProp(Operation operation) {
    super(operation);
    int outputIdx = 0;
  }
}
