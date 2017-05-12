/*
 * op_generator.cpp
 *
 *  Created on: Feb 12, 2017
 *      Author: karl
 */

#include "generator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/logging.h"
#include "utils.h"

using namespace std;

namespace tensorflow {

OpGenerator::OpGenerator(const string& lib_fname) :
  lib_name(GetLastPath(lib_fname)),
  pkg_name(lib_name.substr(0, lib_name.rfind("_op"))),
  file_path(lib_name + "/org/tensorflow/ops/" + pkg_name + "/") {}

OpGenerator::~OpGenerator() {}

void OpGenerator::Run(Env* env, bool include_internal) {
  template_op.Load(env, "op");

  LOG(INFO) << "Generating files for op library <" << lib_name << ">" << endl;
  if (!env->FileExists(file_path).ok()) {
    TF_CHECK_OK(env->RecursivelyCreateDir(file_path));
  }
  OpList ops;
  OpRegistry::Global()->Export(include_internal, &ops);
  WriteOps(ops);
}

void OpGenerator::WriteOps(const OpList& ops) {

  string lib_name_pc = FromUnderscoreToPascalCase(lib_name);
  string lib_name_cc = FromPascalToCamelCase(lib_name_pc);

  for (const auto& op : ops.op()) {
    Template::Params params;

    params["lib_name_pc"] = lib_name_pc;
    params["lib_name_cc"] = lib_name_cc;
    params["pkg_name"] = pkg_name;
    params["imports"] = "";

    WriteOp(op, params);
  }
}

void OpGenerator::WriteOp(const OpDef& op, Template::Params& params) {
  LOG(INFO) << "----- Writing " << pkg_name << "/" << op.name() << endl;

  string op_name_pc = ProtectReservedClasses(op.name(), "Op");
  params["op_name_pc"] = op_name_pc;
  params["op_name_cc"] = ProtectReservedKeywords(FromPascalToCamelCase(op.name()), "Op");
  params["op_name_desc"] = ReplaceAll(op.summary() + "\n\n" + op.description(), "\n", "\n// ");

  TypeEvaluator type_evaluator(op);

  // Write operation inputs
  for (const OpDef_ArgDef& arg : op.input_arg()) {
    Type type = type_evaluator.TypeOf(arg);

    if (ImportType(type, params)) {
      string arg_name_cc = FromUnderscoreToCamelCase(arg.name());
      params["op_inputs"] += ", " + type.JavaType() + ' ' + arg_name_cc;
      params["op_inputs_builder"] += "\n        ";
      if (type_evaluator.ArgIsList(arg)) {
        params["op_inputs_builder"] += ".addInputList(" + arg_name_cc + ".inputList())";
      } else {
        params["op_inputs_builder"] += ".addInput(" + arg_name_cc + ".input())";
      }
    }
  }

  // Write operation outputs
  string output_type, output_name_cc;
  for (const OpDef_ArgDef& arg : op.output_arg()) {
    const Type type = type_evaluator.TypeOf(arg);

    if (ImportType(type, params)) {
      Template::Params output_params(params);

      output_name_cc = FromUnderscoreToCamelCase(arg.name());
      output_type = type.JavaType();

      output_params["output_name_cc"] = output_name_cc;
      output_params["output_type"] = output_type;
      output_params["output_is_array"] = type_evaluator.ArgIsList(arg) ? "[]" : "";

      if (op.output_arg().size() == 1) {
        // have the class implement the output class, and implement
        // the appropriate method if necessary.
        params["op_implements"] = "implements " + output_type;

        if ((output_type == "InputSource" || output_type == "VariableInputSource")
            && output_name_cc != "input") {
          output_params["op_method_name"] = "input";
          params.groups["op_output_setter"].push_back(output_params);

        } else if ((output_type == "InputListSource" || output_type == "VariableListInputSource")
                   && output_name_cc != "inputList") {
          output_params["op_method_name"] = "inputList";
          params.groups["op_output_list_setter"].push_back(output_params);
        }
        params.groups["op_implement_method"].push_back(output_params);
      }
      params.groups["op_outputs"].push_back(output_params);
    }
  }

  // Write operation attributes
  for (const OpDef_AttrDef& attr : op.attr()) {
    const Type type = type_evaluator.TypeOf(attr);

    if (!type.Inferred() && ImportType(type, params)) {
      Template::Params attr_params(params);

      string attr_name_cc = FromUnderscoreToCamelCase(attr.name());
      string attr_name_pc = FromUnderscoreToPascalCase(attr.name());

      attr_params["attr_name"] = attr.name();
      attr_params["attr_name_pc"] = attr_name_pc;
      attr_params["attr_type"] = type.JavaType();

      params.groups["op_attrs"].push_back(attr_params);

      if (!attr.has_default_value()) {
        params["op_params"] += ", " + type.JavaType() + ' ' + attr_name_cc;
        params["op_mandatory_attrs"] += ".with" + attr_name_pc + "(" + attr_name_cc + ")";
      }
    }
  }

  template_op.RenderToFile(ToFileName(op_name_pc),  params);
}

bool OpGenerator::ImportType(const Type& type, Template::Params& params) {
  if (!type.Undefined()) {
    if (!type.JavaImport().empty()) {
      string import = "import " + type.JavaImport() + ";\n";
      if (params["imports"].find(import) == string::npos) {
        params["imports"].append(import);
      }
    }
    return true;
  }
  return false;
}

string OpGenerator::ReplaceAll(const string& in, const string& find, const string& replace) {
  string newString;
  newString.reserve(in.length());  // avoids a few memory allocations

  string::size_type lastPos = 0;
  string::size_type findPos;

  while(string::npos != (findPos = in.find(find, lastPos))) {
    newString.append(in, lastPos, findPos - lastPos);
    newString += replace;
    lastPos = findPos + find.length();
  }

  // Care for the rest after last occurrence
  newString += in.substr(lastPos);
  return newString;
}

} // namespace tensorflow
