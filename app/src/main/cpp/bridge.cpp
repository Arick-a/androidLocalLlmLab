#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <string>
#include <vector>
#include "llama.h"

namespace {

    constexpr char kLogTag[] = "NativeLlm";

    struct ChatMessage {
        std::string role;
        std::string content;
    };

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
            nextPosition_ = 0;
        }

        void requestStop() {
            stopRequested_.store(true);
        }

        void resetStopRequest() {
            stopRequested_.store(false);
        }

        std::vector<llama_token> tokenize(
                const std::string &text,
                bool parseSpecial = false
        ) const {
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
                    parseSpecial
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
                        parseSpecial
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

            if (tokens.size() > llama_n_ctx(context_)) {
                return 0;
            }

            // 基础多轮版每次都用完整历史重建 prompt，因此旧 KV Cache 不能复用。
            llama_memory_clear(llama_get_memory(context_), false);

            // n_batch 是单次 llama_decode 的上限，不是整个 prompt 的上限。
            // 历史消息可能超过 512 Token，因此分批 Prefill，但 position 必须连续。
            const size_t batchSize = llama_n_batch(context_);
            for (size_t start = 0; start < tokens.size(); start += batchSize) {
                if (stopRequested_.load()) {
                    return 0;
                }

                const size_t count = std::min(batchSize, tokens.size() - start);
                llama_batch batch = llama_batch_init(static_cast<int32_t>(count), 0, 1);

                for (size_t offset = 0; offset < count; ++offset) {
                    const size_t index = start + offset;
                    batch.token[offset] = tokens[index];
                    batch.pos[offset] = static_cast<llama_pos>(index);
                    batch.n_seq_id[offset] = 1;
                    batch.seq_id[offset][0] = 0;

                    // 只有完整 prompt 的最后一个 Token 需要 Logits。
                    batch.logits[offset] = index == tokens.size() - 1;
                }

                batch.n_tokens = static_cast<int32_t>(count);
                const int32_t decodeResult = llama_decode(context_, batch);
                llama_batch_free(batch);

                if (decodeResult != 0) {
                    return 0;
                }
            }

            nextPosition_ = static_cast<llama_pos>(tokens.size());
            return static_cast<int>(tokens.size());
        }

        std::string generate(
                const std::vector<ChatMessage> &history,
                int maxTokens,
                const std::function<bool(const std::string &, int)> &onPrompt,
                const std::function<bool(const std::string &)> &onToken
        ) {
            if (model_ == nullptr || context_ == nullptr || history.empty() || maxTokens <= 0) {
                return {};
            }

            if (stopRequested_.load()) {
                return {};
            }

            // 模板来自当前 GGUF 的 tokenizer.chat_template 元数据，而不是写死 Qwen 的
            // ChatML。不同 Instruct 模型（Qwen、Llama、Mistral 等）可以各自决定
            // user / assistant 消息应如何组织。
            const char *chatTemplate = llama_model_chat_template(model_, nullptr);
            if (chatTemplate == nullptr) {
                __android_log_print(
                        ANDROID_LOG_WARN,
                        kLogTag,
                        "The loaded GGUF has no chat template"
                );
                return {};
            }

            // 保持字符串所有权，再创建 llama.cpp 所需的 role/content 指针数组。
            // 这样每轮都能把 user 与 assistant 的完整历史交给同一个模型模板。
            std::vector<llama_chat_message> messages;
            messages.reserve(history.size());
            size_t totalCharacters = 0;
            for (const ChatMessage &message : history) {
                if (message.role.empty() || message.content.empty()) {
                    return {};
                }
                messages.push_back({message.role.c_str(), message.content.c_str()});
                totalCharacters += message.role.size() + message.content.size();
            }

            // addAssistant=true 表示结尾追加 assistant 的回复起始标记。
            std::vector<char> formattedPrompt(totalCharacters * 2 + 128);
            int32_t promptLength = llama_chat_apply_template(
                    chatTemplate,
                    messages.data(),
                    messages.size(),
                    true,
                    formattedPrompt.data(),
                    static_cast<int32_t>(formattedPrompt.size())
            );

            if (promptLength < 0) {
                return {};
            }

            if (promptLength > static_cast<int32_t>(formattedPrompt.size())) {
                formattedPrompt.resize(static_cast<size_t>(promptLength));
                promptLength = llama_chat_apply_template(
                        chatTemplate,
                        messages.data(),
                        messages.size(),
                        true,
                        formattedPrompt.data(),
                        static_cast<int32_t>(formattedPrompt.size())
                );
            }

            if (promptLength <= 0) {
                return {};
            }

            const std::string prompt(
                    formattedPrompt.data(),
                    static_cast<size_t>(promptLength)
            );

            const std::vector<llama_token> promptTokens = tokenize(prompt, true);
            // Token 数由当前 GGUF 的 tokenizer 得到，因此包含模型模板标记与特殊 Token。
            // 它才是这轮 Prompt 占用 Context 的真实计量，而不是中文字符数。
            if (!onPrompt(prompt, static_cast<int>(promptTokens.size()))) {
                return {};
            }

            if (prefill(promptTokens) != static_cast<int>(promptTokens.size())) {
                return {};
            }

            if (stopRequested_.load()) {
                return {};
            }

            const llama_vocab *vocab = llama_model_get_vocab(model_);
            if (vocab == nullptr) {
                return {};
            }

            llama_sampler *sampler = llama_sampler_init_greedy();
            if (sampler == nullptr) {
                return {};
            }

            std::string answer;
            for (int index = 0; index < maxTokens; ++index) {
                // JNI 的停止请求可以在 Kotlin 主线程写入；atomic 让 Native 推理线程
                // 无锁地安全读取。最迟在当前一个 Token 完成后退出。
                if (stopRequested_.load()) {
                    break;
                }

                const llama_token token = llama_sampler_sample(sampler, context_, -1);
                if (llama_vocab_is_eog(vocab, token)) {
                    break;
                }

                const std::string piece = tokenToPiece(token, false);
                answer += piece;

                // Native 每采样出一个 Token，就把对应文字片段推回 Kotlin。
                // 这样 Compose 不必等完整回答生成完才更新页面。
                if (!piece.empty() && !onToken(piece)) {
                    answer.clear();
                    break;
                }

                if (stopRequested_.load()) {
                    break;
                }

                llama_sampler_accept(sampler, token);

                if (!decodeToken(token)) {
                    answer.clear();
                    break;
                }
            }

            llama_sampler_free(sampler);
            return answer;
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

        std::string tokenToPiece(llama_token token, bool special = true) const {
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
                    special
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
                        special
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

        bool decodeToken(llama_token token) {
            if (context_ == nullptr) {
                return false;
            }

            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.token[0] = token;
            batch.pos[0] = nextPosition_;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            batch.n_tokens = 1;

            const int32_t decodeResult = llama_decode(context_, batch);
            llama_batch_free(batch);

            if (decodeResult != 0) {
                return false;
            }

            ++nextPosition_;
            return true;
        }

    private:
        llama_model *model_ = nullptr;
        llama_context *context_ = nullptr;
        llama_pos nextPosition_ = 0;
        std::atomic_bool stopRequested_ = false;
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
JNIEXPORT void JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeRequestStop(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle) {
    auto *runtime = toRuntime(handle);
    if (runtime != nullptr) {
        runtime->requestStop();
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeResetStopRequest(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle) {
    auto *runtime = toRuntime(handle);
    if (runtime != nullptr) {
        runtime->resetStopRequest();
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

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeGenerate(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jobjectArray roles,
        jobjectArray contents,
        jint maxTokens,
        jobject generationCallback) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr || roles == nullptr || contents == nullptr || maxTokens <= 0 || generationCallback == nullptr) {
        return env->NewStringUTF("");
    }

    const jsize messageCount = env->GetArrayLength(roles);
    if (messageCount == 0 || messageCount != env->GetArrayLength(contents)) {
        return env->NewStringUTF("");
    }

    std::vector<ChatMessage> history;
    history.reserve(static_cast<size_t>(messageCount));
    for (jsize index = 0; index < messageCount; ++index) {
        auto *role = static_cast<jstring>(env->GetObjectArrayElement(roles, index));
        auto *content = static_cast<jstring>(env->GetObjectArrayElement(contents, index));
        if (role == nullptr || content == nullptr) {
            return env->NewStringUTF("");
        }

        const char *roleChars = env->GetStringUTFChars(role, nullptr);
        const char *contentChars = env->GetStringUTFChars(content, nullptr);
        if (roleChars == nullptr || contentChars == nullptr) {
            if (roleChars != nullptr) {
                env->ReleaseStringUTFChars(role, roleChars);
            }
            if (contentChars != nullptr) {
                env->ReleaseStringUTFChars(content, contentChars);
            }
            return env->NewStringUTF("");
        }

        history.push_back({roleChars, contentChars});
        env->ReleaseStringUTFChars(role, roleChars);
        env->ReleaseStringUTFChars(content, contentChars);
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
    }

    jclass callbackClass = env->GetObjectClass(generationCallback);
    jmethodID onPromptMethod = callbackClass == nullptr
            ? nullptr
            : env->GetMethodID(callbackClass, "onPrompt", "(Ljava/lang/String;I)V");
    jmethodID onTokenMethod = callbackClass == nullptr
            ? nullptr
            : env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
    if (onPromptMethod == nullptr || onTokenMethod == nullptr) {
        env->ExceptionClear();
        return env->NewStringUTF("");
    }

    const std::string answer = runtime->generate(
            history,
            maxTokens,
            [env, generationCallback, onPromptMethod](const std::string &prompt, int tokenCount) {
                jstring kotlinPrompt = env->NewStringUTF(prompt.c_str());
                if (kotlinPrompt == nullptr) {
                    return false;
                }

                env->CallVoidMethod(
                        generationCallback,
                        onPromptMethod,
                        kotlinPrompt,
                        static_cast<jint>(tokenCount)
                );
                env->DeleteLocalRef(kotlinPrompt);

                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                    return false;
                }
                return true;
            },
            [env, generationCallback, onTokenMethod](const std::string &piece) {
                jstring kotlinPiece = env->NewStringUTF(piece.c_str());
                if (kotlinPiece == nullptr) {
                    return false;
                }

                env->CallVoidMethod(generationCallback, onTokenMethod, kotlinPiece);
                env->DeleteLocalRef(kotlinPiece);

                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                    return false;
                }
                return true;
            }
    );
    return env->NewStringUTF(answer.c_str());
}
