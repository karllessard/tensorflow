/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
package org.tensorflow;

import java.io.Closeable;
import java.nio.charset.Charset;

final class EagerOperationBuilder implements OperationBuilder, Closeable {

  @Override
  public Operation build() {
    long[] outputNativeHandles = execute(nativeHandle);
    EagerOperation op = new EagerOperation(nativeHandle, outputNativeHandles, type, name);
    // From now on, the EagerOperation will carry the op native handle, so detach it from this builder
    session.allocatedResources().remove(this);
    nativeHandle = 0L;
    return op;
  }

  @Override
  public EagerOperationBuilder addInput(Output<?> input) {
    addInput(nativeHandle, input.getNativeHandle());
    return this;
  }

  @Override
  public OperationBuilder addInputList(Output<?>[] inputs) {
    long[] inputHandles = new long[inputs.length];
    for (int i = 0; i < inputs.length; ++i) {
      inputHandles[i] = inputs[i].getNativeHandle();
    }
    addInputList(nativeHandle, inputHandles);
    return this;
  }

  @Override
  public OperationBuilder addControlInput(Operation control) {
    // FIXME (karllessard) maybe we can support them by executing the control operation right away?
    throw new UnsupportedOperationException("Control inputs are not supported in eager mode");
  }

  @Override
  public OperationBuilder setDevice(String device) {
    setDevice(nativeHandle, device);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, String value) {
    return setAttr(name, value.getBytes(Charset.forName("UTF-8")));
  }

  @Override
  public OperationBuilder setAttr(String name, String[] values) {
    Charset utf8 = Charset.forName("UTF-8");
    Object[] objects = new Object[values.length];
    for (int i = 0; i < values.length; ++i) {
      objects[i] = values[i].getBytes(utf8);
    }
    setAttrStringList(nativeHandle, name, values);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, byte[] values) {
    setAttrString(nativeHandle, name, values);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, long value) {
    setAttrInt(nativeHandle, name, value);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, long[] values) {
    setAttrIntList(nativeHandle, name, values);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, float value) {
    setAttrFloat(nativeHandle, name, value);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, float[] values) {
    setAttrFloatList(nativeHandle, name, values);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, boolean value) {
    setAttrBool(nativeHandle, name, value);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, boolean[] values) {
    setAttrBoolList(nativeHandle, name, values);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, DataType value) {
    setAttrType(nativeHandle, name, value.c());
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, DataType[] values) {
    int[] c = new int[values.length];
    for (int i = 0; i < values.length; ++i) {
      c[i] = values[i].c();
    }
    setAttrTypeList(nativeHandle, name, c);
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, Tensor<?> value) {
    setAttrTensor(nativeHandle, name, value.getNativeHandle());
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, Tensor<?>[] values) {
    // TODO (karllessard) could be added to the C API if really needed
    throw new UnsupportedOperationException("Tensor list attributes are not supported in eager mode");
  }

  @Override
  public OperationBuilder setAttr(String name, Shape value) {
    setAttrShape(nativeHandle, name, value.asArray(), value.numDimensions());
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, Shape[] values) {
    int[] numDimensions = new int[values.length];
    int totalNumDimensions = 0;
    for (int idx = 0; idx < values.length; ++idx) {
      int n = values[idx].numDimensions();
      numDimensions[idx] = n;
      if (n > 0) {
        totalNumDimensions += n;
      }
    }
    // Flatten the shapes into a single array to avoid too much overhead in the
    // native part
    long[] shapes = new long[totalNumDimensions];
    int shapeIdx = 0;
    for (Shape shape : values) {
      if (shape.numDimensions() > 0) {
        for (long dim : shape.asArray()) {
          shapes[shapeIdx++] = dim;
        }
      }
    }
    setAttrShapeList(nativeHandle, name, shapes, numDimensions);
    return this;
  }

  @Override
  public void close() {
    if (nativeHandle != 0L) {
      delete(nativeHandle);
      nativeHandle = 0L;
    }
  }

  EagerOperationBuilder(EagerSession session, long nativeHandle, String type, String name) {
    this.session = session;
    this.type = type;
    this.name = name;
    this.nativeHandle = nativeHandle;
  }

  private final EagerSession session;
  private final String type;
  private final String name;
  private long nativeHandle;
  
  private static native void delete(long opHandle);
  
  private static native long[] execute(long opHandle);

  private static native void addInput(long opHandle, long tensorHandle);

  private static native void addInputList(long opHandle, long[] tensorHandles);

  private static native void setDevice(long opHandle, String device);

  private static native void setAttrString(long opHandle, String name, byte[] value);

  private static native void setAttrStringList(long opHandle, String name, Object[] value);

  private static native void setAttrInt(long opHandle, String name, long value);

  private static native void setAttrIntList(long opHandle, String name, long[] values);

  private static native void setAttrFloat(long opHandle, String name, float value);

  private static native void setAttrFloatList(long opHandle, String name, float[] values);

  private static native void setAttrBool(long opHandle, String name, boolean value);

  private static native void setAttrBoolList(long opHandle, String name, boolean[] values);

  private static native void setAttrType(long opHandle, String name, int type);

  private static native void setAttrTypeList(long opHandle, String name, int[] types);

  private static native void setAttrTensor(long opHandle, String name, long tensorHandle);

  private static native void setAttrShape(long opHandle, String name, long[] shape, int numDims);

  private static native void setAttrShapeList(long opHandle, String name, long[] shapes, int[] numDims);
}
