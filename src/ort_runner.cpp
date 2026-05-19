#include "ort_runner.hpp"

#include <godot_cpp/core/class_db.hpp>

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/project_settings.hpp>

#include "onnx_mingw_overrides.h"

#include"ort_session.hpp"


#include<iostream>
#include <vector>
#include <cstdlib>

#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

using namespace godot;

OnnxRunner *OnnxRunner::singleton = nullptr;

static String _get_godot_path(String pathName)
{
	auto settings = ProjectSettings::get_singleton();
	String global_path = settings->globalize_path(pathName);
	return global_path;
}

void OnnxRunner::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("load_model"), &OnnxRunner::load_model);
	ClassDB::bind_method(D_METHOD("get_available_providers"), &OnnxRunner::get_available_providers);
}

OnnxRunner *OnnxRunner::get_singleton()
{
	return singleton;
}

OnnxRunner::OnnxRunner()
{
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
	initialized_api = false;
	UtilityFunctions::print("DroneDetector: ONNX Runner Initializing...");
}

OnnxRunner::~OnnxRunner()
{
	ERR_FAIL_COND(singleton != this);
	delete env;
	singleton = nullptr;
}

void log_fn (void *param, OrtLoggingLevel severity, const char *category, const char *logid, const char *code_location, const char *message)
{
	if(message!=NULL){
		UtilityFunctions::print("ONNX: ", message);
	}
}

void OnnxRunner::_init_api()
{
	if(!initialized_api){
		#ifdef ORT_API_MANUAL_INIT
		Ort::InitApi();
		#endif

		env=new Ort::Env();
		initialized_api=true;
		
		PackedStringArray providers = get_available_providers();
		UtilityFunctions::print("DroneDetector: Available ONNX Providers: ", providers);
	}
}

PackedStringArray OnnxRunner::get_available_providers() {
	PackedStringArray providers;
	const auto& api = Ort::GetApi();
	char** provider_names;
	int num_providers;
	if (api.GetAvailableProviders(&provider_names, &num_providers) == nullptr) {
		for (int i = 0; i < num_providers; i++) {
			providers.append(provider_names[i]);
		}
		api.ReleaseAvailableProviders(provider_names, num_providers);
	}
	return providers;
}

OnnxSession* OnnxRunner::load_model(String model_source_local )
{
	String model_source = _get_godot_path(model_source_local);
	ERR_FAIL_COND_V_MSG(!FileAccess::file_exists(model_source),NULL, vformat("Couldn't find file: %s",model_source_local));

	_init_api();
	const auto& api = Ort::GetApi();

	Ort::SessionOptions session_options;
	// IntraOpNumThreads(1) is often faster for mobile/nano models when using EPs
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	#ifdef __APPLE__
	bool use_coreml = true;
	auto settings = ProjectSettings::get_singleton();
	if (settings->has_setting("onnx/use_coreml")) {
		use_coreml = (bool)settings->get_setting("onnx/use_coreml");
	}
	if (std::getenv("DISABLE_COREML") != nullptr) {
		use_coreml = false;
	}

	if (use_coreml) {
		// Enable CoreML EP on macOS for hardware acceleration (ANE/GPU/CPU)
		uint32_t coreml_flags = 0;
		coreml_flags |= COREML_FLAG_CREATE_MLPROGRAM;
		// coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE; // Optional: Force Neural Engine
		
		OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(static_cast<OrtSessionOptions*>(session_options), coreml_flags);
		if (status != nullptr) {
			UtilityFunctions::printerr("DroneDetector: CoreML EP failed to initialize: ", api.GetErrorMessage(status), ". Falling back to CPU.");
			api.ReleaseStatus(status);
		} else {
			UtilityFunctions::print("DroneDetector: CoreML Execution Provider successfully appended.");
		}
	} else {
		UtilityFunctions::print("DroneDetector: CoreML Execution Provider disabled. Falling back to CPU.");
	}
	#endif
	
	#ifdef _WIN32
	const wchar_t* model_path = model_source.wide_string().get_data();
	#else
	CharString model_source_ascii = model_source.ascii();
	const char* model_path = model_source_ascii.get_data();
	#endif
	
	UtilityFunctions::print("DroneDetector: Loading model from: ", model_source);
	
	{
		OrtExceptionCatcher catcher;
		Ort::Session *session=new Ort::Session(*env, model_path, session_options);
		if(catcher.HasError()){
			ERR_FAIL_V_MSG(NULL, vformat("Error creating onnx model session: %s (%d)",catcher.GetErrorString(),catcher.GetErrorCode()));
		}
		UtilityFunctions::print("DroneDetector: ONNX Session created successfully.");

		OnnxSession* newSession= memnew(OnnxSession);
		newSession->connectOnnxSession(session);
		return newSession;
	}
}

void OnnxRunner::hello_singleton()
{
	UtilityFunctions::print("Hello GDExtension Singleton!");
}
