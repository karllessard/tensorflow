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
package org.tensorflow.op.io;

import java.util.Arrays;
import java.util.Iterator;
import org.tensorflow.DataType;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;

public final class RestoreV2 extends PrimitiveOp implements Iterable<Operand<DataType>> {
  
  /**
   * Factory method to create a class to wrap a new RestoreV2 operation to the graph.
   * 
   * @param scope Current graph scope
   * @param prefix
   * @param tensorNames
   * @param shapeAndSlices
   * @param dtypes
   * @return a new instance of RestoreV2
   **/
  public static RestoreV2 create(Scope scope, Operand<String> prefix, Operand<String> tensorNames, Operand<String> shapeAndSlices, List<DataType> dtypes) {
    OperationBuilder opBuilder = scope.graph().opBuilder("RestoreV2", scope.makeOpName("RestoreV2"));
    opBuilder.addInput(prefix.asOutput());
    opBuilder.addInput(tensorNames.asOutput());
    opBuilder.addInput(shapeAndSlices.asOutput());
    opBuilder.setAttr("dtypes", dtypes.toArray(new DataType[dtypes.size()]));
    return new RestoreV2(opBuilder.build());
  }
  
  public List<Output<DataType>> tensors() {
    return tensors;
  }
  
  @Override
  @SuppressWarnings({"rawtypes", "unchecked"})
  public Iterator<Operand<DataType>> iterator() {
    return (Iterator) tensors.iterator();
  }
  
  private List<Output<DataType>> tensors;
  
  /**
   * @param operation
   **/
  @SuppressWarnings("unchecked")
  private RestoreV2(Operation operation) {
    super(operation);
    int outputIdx = 0;
    int tensorsLength = operation.outputListLength("tensors");
    tensors = Arrays.asList((Output<DataType>[])operation.outputList(outputIdx, tensorsLength));
    outputIdx += tensorsLength;
  }
}
