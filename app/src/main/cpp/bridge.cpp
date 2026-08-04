#include <jni.h>
#include <android/log.h>
#include <string>

#include "llama.h"

namespace {

    constexpr char kLogTag[] = "NativeLlm";

    class NativeRuntime {
    public:
        NativeRuntime() {
            llama_backend_init();
            __android_log_print(
                    ANDROID_LOG_INFO,
                    kLogTag,
                    "llama.cpp backend initialized"
            );
        }

        ~NativeRuntime() {
            unloadModel();
            llama_backend_free();
            __android_log_print(
                    ANDROID_LOG_INFO,
                    kLogTag,
                    "llama.cpp backend released"
            );
        }

        bool loadModel(const std::string &modelPath) {
            if (model_ != nullptr) {
                return false;
            }

            const llama_model_params params = llama_model_default_params();

            model_ = llama_model_load_from_file(
                    modelPath.c_str(),
                    params
            );

            return model_ != nullptr;
        }

        void unloadModel() {
            if (model_ == nullptr) {
                return;
            }

            llama_model_free(model_);
            model_ = nullptr;
        }

    private:
        llama_model *model_ = nullptr;
    };
}

NativeRuntime *toRuntime(jlong handle) {
    return reinterpret_cast<NativeRuntime *>(handle);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeCreateRuntime(
        JNIEnv *env,
        jobject thiz) {
    auto *runtime = new NativeRuntime();
    return reinterpret_cast<jlong>(runtime);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeReleaseRuntime(
        JNIEnv *env,
        jobject thiz,
        jlong handle) {
    if (handle == 0L) {
        return;
    }

    delete toRuntime(handle);
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeLoadModel(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jstring modelPath) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr || modelPath == nullptr) {
        return JNI_FALSE;
    }

    const char *pathChars = env->GetStringUTFChars(modelPath, nullptr);
    if (pathChars == nullptr) {
        return JNI_FALSE;
    }

    const bool loaded = runtime->loadModel(pathChars);

    env->ReleaseStringUTFChars(modelPath, pathChars);
    return loaded ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeUnloadModel(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle) {
    auto *runtime = toRuntime(handle);
    if (runtime != nullptr) {
        runtime->unloadModel();
    }
}