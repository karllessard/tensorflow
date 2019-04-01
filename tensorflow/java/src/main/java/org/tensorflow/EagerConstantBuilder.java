package org.tensorflow;

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

final class EagerConstantBuilder implements OperationBuilder {

  @Override
  public Operation build() {
    long tensorHandle = allocateTensorHandle(tensor.getNativeHandle());
    return new EagerConstant(session, tensor, tensorHandle, name);
  }

  @Override
  public OperationBuilder setAttr(String name, Tensor<?> value) {
    this.tensor = value;
    value.attachTo(session);
    return this;
  }

  @Override
  public OperationBuilder addInput(Output<?> input) {
    return this;
  }

  @Override
  public OperationBuilder addInputList(Output<?>[] inputs) {
    return this;
  }

  @Override
  public OperationBuilder addControlInput(Operation control) {
    return this;
  }

  @Override
  public OperationBuilder setDevice(String device) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, String value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, byte[] value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, long value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, long[] value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, float value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, float[] value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, boolean value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, boolean[] value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, DataType value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, DataType[] value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, Tensor<?>[] value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, Shape value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, Shape[] value) {
    return this;
  }

  @Override
  public OperationBuilder setAttr(String name, String[] value) {
    return this;
  }
  
  EagerConstantBuilder(EagerSession session, String name) {
    this.session = session;
    this.name = name;
  }

  private final EagerSession session;
  private final String name;
  private Tensor<?> tensor;

  private static native long allocateTensorHandle(long tensorHandle);
}
