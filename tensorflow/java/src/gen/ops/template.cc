/*
 * template.cpp
 *
 *  Created on: Feb 12, 2017
 *      Author: karl
 */
#include <regex>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "template.h"

#define TEMPLATE_PATH "/home/karl/dev/projects/tf/tensorflow/tensorflow/java/src/gen/ops/templates/" // FIXME
#define TEMPLATE_EXT ".tmpl"

using namespace std;

namespace tensorflow {

Template::Template() {
  this->env = NULL;
}

Template::Template(Env* env, const string& template_src) {
  this->env = env;
  this->template_src = template_src;
}

Template::~Template() {}

void Template::Load(Env* env, const string& name) {
  this->env = env;
  this->name = name;
  TF_CHECK_OK(ReadFileToString(env, TEMPLATE_PATH + name + TEMPLATE_EXT, &template_src));
}

string Template::Render(const Params& params) {
  StringWriter writer;
  RenderWith(writer, params);
  return writer.str();
}

void Template::RenderToFile(const string& fname, const Params& params) {
  FileWriter writer(env, fname);
  RenderWith(writer, params);
}

void Template::RenderWith(Writer& writer, const Params& params) {
  size_t pos = 0;
  while (pos < template_src.size()) {
    size_t next_param_pos = template_src.find("${", pos);
    writer.Append(template_src.substr(pos, next_param_pos - pos));
    if (next_param_pos == string::npos) {
      break; // end of template
    }
    pos = ProcessParam(writer, next_param_pos, params);
  }
}

size_t Template::ProcessParam(Writer& writer, size_t param_pos, const Params& params) {
  size_t pos = param_pos += 2;
  regex delim("\\}|\\[\n");
  smatch match;

  string partial_src = template_src.substr(pos);
  regex_search(partial_src, match, delim);
  string param_name = partial_src.substr(0, match.position());
  pos += match.position();

  if (match[0] == '}') {
    map<const string, string>::const_iterator param = params.find(param_name);
    if (param != params.end()) {
      writer.Append(param->second);
    }
    return pos + 1;
  }
  return ProcessInnerTemplateParam(writer, param_name, pos + 2, params);
}

size_t Template::ProcessInnerTemplateParam(Writer& writer, const string& param_name, size_t src_pos, const Params& params) {
  //if (template_src.substr(src_pos, 2) != "[\n") {
  //  LOG(ERROR) << "Unsupported inner template: " << src_pos << endl;
  //  return src_pos; // FIXME unsupported!
  //}
  size_t src_end_pos = template_src.find("]}\n", src_pos);
  Template inner_template(env, template_src.substr(src_pos, src_end_pos - src_pos));

  map<const string, list<Params>>::const_iterator param = params.groups.find(param_name);
  if (param != params.groups.end()) {
    list<Params> params_list = param->second;
    for (list<Params>::const_iterator it = params_list.begin(); it != params_list.end(); ++it) {
      inner_template.RenderWith(writer, *it);
    }
  }
  return src_end_pos + 3;
}

}
