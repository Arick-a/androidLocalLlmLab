#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
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
            releaseContext();
            if (model_ == nullptr) {
                return;
            }

            llama_model_free(model_);
            model_ = nullptr;
        }

        int createContext(uint32_t requestedNctx) {
            if (model_ == nullptr || context_ != nullptr) {
                return 0;
            }

            llama_context_params params = llama_context_default_params();
            params.n_ctx = requestedNctx;
            params.n_batch = 512;
            params.n_ubatch = 512;

            context_ = llama_init_from_model(model_, params);
            if (context_ == nullptr) {
                return 0;
            }

            return static_cast<int>(llama_n_ctx(context_));
        }

        void releaseContext() {
            if (context_ == nullptr) {
                return;
            }

            llama_free(context_);
            context_ = nullptr;
        }

        std::vector<llama_token> tokenize(const std::string &text) const {
            if (model_ == nullptr) {
                return {};
            }

            const llama_vocab *vocab = llama_model_get_vocab(model_);
            if (vocab == nullptr) {
                return {};
            }

            std::vector<llama_token> tokens(text.size() + 8);

            int32_t tokenCount = llama_tokenize(
                    vocab,
                    text.c_str(),
                    static_cast<int32_t>(text.size()),
                    tokens.data(),
                    static_cast<int32_t>(tokens.size()),
                    false,
                    false
            );

            // llama.cpp 用负数告知“当前 Token 缓冲区不够，实际需要多少个”。
            if (tokenCount < 0) {
                tokens.resize(static_cast<size_t>(-tokenCount));

                tokenCount = llama_tokenize(
                        vocab,
                        text.c_str(),
                        static_cast<int32_t>(text.size()),
                        tokens.data(),
                        static_cast<int32_t>(tokens.size()),
                        false,
                        false
                );
            }

            if (tokenCount < 0) {
                return {};
            }

            tokens.resize(static_cast<size_t>(tokenCount));
            return tokens;
        }

        int prefill(const std::vector<llama_token> &tokens) {
            if (context_ == nullptr || tokens.empty()) {
                return 0;
            }

            // 当前实验一次最多提交 n_batch 个 Token。
            if (tokens.size() > llama_n_batch(context_)) {
                return 0;
            }

            // 本实验每次都是一段全新的固定文本，因此先清除旧 KV Cache。
            llama_memory_clear(llama_get_memory(context_), false);

            llama_batch batch = llama_batch_init(
                    static_cast<int32_t>(tokens.size()),
                    0,
                    1
            );

            for (size_t index = 0; index < tokens.size(); ++index) {
                batch.token[index] = tokens[index];
                batch.pos[index] = static_cast<llama_pos>(index);
                batch.n_seq_id[index] = 1;
                batch.seq_id[index][0] = 0;

                // 只有最后一个 Token 需要 Logits。
                // 下一步 Sampling 会从这行 Logits 中选择下一个 Token。
                batch.logits[index] = index == tokens.size() - 1;
            }

            batch.n_tokens = static_cast<int32_t>(tokens.size());

            const int32_t decodeResult = llama_decode(context_, batch);
            llama_batch_free(batch);

            if (decodeResult != 0) {
                return 0;
            }

            return static_cast<int>(tokens.size());
        }

        int sampleNextToken() const {
            if (context_ == nullptr) {
                return -1;
            }

            // Greedy = 始终选择当前概率最大的 Token。
            // 它适合当前对照实验：相同输入必然得到相同结果。
            llama_sampler *sampler = llama_sampler_init_greedy();
            if (sampler == nullptr) {
                return -1;
            }

            const llama_token token = llama_sampler_sample(
                    sampler,
                    context_,
                    -1
            );

            llama_sampler_free(sampler);
            return token;
        }

        std::string tokenToPiece(llama_token token) const {
            if (model_ == nullptr) {
                return {};
            }

            const llama_vocab *vocab = llama_model_get_vocab(model_);
            if (vocab == nullptr) {
                return {};
            }

            std::vector<char> buffer(32);

            int32_t pieceLength = llama_token_to_piece(
                    vocab,
                    token,
                    buffer.data(),
                    static_cast<int32_t>(buffer.size()),
                    0,
                    true
            );

            // 返回负数说明 buffer 不够，负数绝对值是所需字节数。
            if (pieceLength < 0) {
                buffer.resize(static_cast<size_t>(-pieceLength));

                pieceLength = llama_token_to_piece(
                        vocab,
                        token,
                        buffer.data(),
                        static_cast<int32_t>(buffer.size()),
                        0,
                        true
                );
            }

            if (pieceLength <= 0) {
                return {};
            }

            return {
                    buffer.data(),
                    static_cast<size_t>(pieceLength)
            };
        }

    private:
        llama_model *model_ = nullptr;
        llama_context *context_ = nullptr;
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

extern "C"
JNIEXPORT jint JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeCreateContext(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jint requestedNctx) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr || requestedNctx <= 0) {
        return 0;
    }

    return runtime->createContext(
            static_cast<uint32_t>(requestedNctx)
    );
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeReleaseContext(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle) {
    auto *runtime = toRuntime(handle);
    if (runtime != nullptr) {
        runtime->releaseContext();
    }
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeTokenize(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jstring text) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr || text == nullptr) {
        return env->NewIntArray(0);
    }

    const char *textChars = env->GetStringUTFChars(text, nullptr);
    if (textChars == nullptr) {
        return env->NewIntArray(0);
    }

    const std::vector<llama_token> tokens = runtime->tokenize(textChars);
    env->ReleaseStringUTFChars(text, textChars);

    jintArray result = env->NewIntArray(
            static_cast<jsize>(tokens.size())
    );

    if (result == nullptr || tokens.empty()) {
        return result;
    }

    std::vector<jint> tokenIds(tokens.begin(), tokens.end());
    env->SetIntArrayRegion(
            result,
            0,
            static_cast<jsize>(tokenIds.size()),
            tokenIds.data()
    );

    return result;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativePrefill(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jintArray tokenIds) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr || tokenIds == nullptr) {
        return 0;
    }

    const jsize tokenCount = env->GetArrayLength(tokenIds);
    if (tokenCount == 0) {
        return 0;
    }

    jint *rawTokenIds = env->GetIntArrayElements(tokenIds, nullptr);
    if (rawTokenIds == nullptr) {
        return 0;
    }

    std::vector<llama_token> tokens(
            rawTokenIds,
            rawTokenIds + tokenCount
    );

    // C++ 已复制数组，不需要把任何修改回写到 Kotlin IntArray。
    env->ReleaseIntArrayElements(tokenIds, rawTokenIds, JNI_ABORT);

    return runtime->prefill(tokens);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeSampleNextToken(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr) {
        return -1;
    }

    return runtime->sampleNextToken();
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeTokenToPiece(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jint tokenId) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr) {
        return env->NewStringUTF("");
    }

    const std::string piece = runtime->tokenToPiece(
            static_cast<llama_token>(tokenId)
    );

    return env->NewStringUTF(piece.c_str());
}
