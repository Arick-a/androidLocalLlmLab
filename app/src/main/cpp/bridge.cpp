#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <chrono>
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

    struct SamplingConfig {
        float temperature;
        int topK;
        float topP;
        float minP;
        float repeatPenalty;
        uint32_t seed;
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

        std::string modelDescription() const {
            if (model_ == nullptr) {
                return {};
            }
            std::vector<char> buffer(256);
            const int32_t length = llama_model_desc(model_, buffer.data(), buffer.size());
            if (length <= 0) {
                return {};
            }
            return {buffer.data(), static_cast<size_t>(length)};
        }

        std::string modelMetadata(const char *key) const {
            if (model_ == nullptr || key == nullptr) {
                return {};
            }
            std::vector<char> buffer(256);
            const int32_t length = llama_model_meta_val_str(
                    model_,
                    key,
                    buffer.data(),
                    buffer.size()
            );
            if (length <= 0) {
                return {};
            }
            return {buffer.data(), static_cast<size_t>(length)};
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
            clearCachedTokens();
        }

        // 不释放模型权重，只清除当前会话在 Native 中累积的 KV Cache。
        void resetContext() {
            if (context_ == nullptr) {
                return;
            }

            llama_memory_clear(llama_get_memory(context_), false);
            clearCachedTokens();
        }

        int setThreads(int generationThreads, int batchThreads) {
            if (context_ == nullptr || generationThreads <= 0 || batchThreads <= 0) {
                return 0;
            }

            llama_set_n_threads(context_, generationThreads, batchThreads);
            return llama_n_threads(context_);
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

        // 使用当前 GGUF 自带的 Chat Template 生成 Prompt。计数和真正推理必须复用
        // 这条路径，否则 Kotlin 按 Token 预算裁剪时会与 Native 实际 Prefill 不一致。
        std::string formatChatPrompt(const std::vector<ChatMessage> &history) const {
            if (model_ == nullptr || history.empty()) {
                return {};
            }

            const char *chatTemplate = llama_model_chat_template(model_, nullptr);
            if (chatTemplate == nullptr) {
                __android_log_print(
                        ANDROID_LOG_WARN,
                        kLogTag,
                        "The loaded GGUF has no chat template"
                );
                return {};
            }

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

            return {
                    formattedPrompt.data(),
                    static_cast<size_t>(promptLength)
            };
        }

        int countChatPromptTokens(const std::vector<ChatMessage> &history) const {
            const std::string prompt = formatChatPrompt(history);
            if (prompt.empty()) {
                return 0;
            }
            return static_cast<int>(tokenize(prompt, true).size());
        }

        int prefill(const std::vector<llama_token> &tokens, size_t startIndex) {
            if (context_ == nullptr || tokens.empty()) {
                return 0;
            }

            if (tokens.size() > llama_n_ctx(context_)) {
                return 0;
            }

            if (startIndex > tokens.size()) {
                return 0;
            }

            // n_batch 是单次 llama_decode 的上限，不是整个 prompt 的上限。
            // 历史消息可能超过 512 Token，因此分批 Prefill，但 position 必须连续。
            const size_t batchSize = llama_n_batch(context_);
            for (size_t start = startIndex; start < tokens.size(); start += batchSize) {
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
                const SamplingConfig &samplingConfig,
                const std::function<bool(const std::string &, int)> &onPrompt,
                const std::function<bool(const std::string &)> &onToken,
                const std::function<void(int, int, int64_t, int64_t, int64_t, int64_t)> &onMetrics
        ) {
            if (model_ == nullptr || context_ == nullptr || history.empty() || maxTokens <= 0) {
                return {};
            }

            if (stopRequested_.load()) {
                return {};
            }

            using Clock = std::chrono::steady_clock;
            const auto generationStartedAt = Clock::now();
            const std::string prompt = formatChatPrompt(history);
            if (prompt.empty()) {
                return {};
            }

            const std::vector<llama_token> promptTokens = tokenize(prompt, true);
            // Token 数由当前 GGUF 的 tokenizer 得到，因此包含模型模板标记与特殊 Token。
            // 它才是这轮 Prompt 占用 Context 的真实计量，而不是中文字符数。
            if (!onPrompt(prompt, static_cast<int>(promptTokens.size()))) {
                return {};
            }

            const size_t reusedPromptTokenCount = commonPrefixLength(cachedTokens_, promptTokens);
            const bool canReuseCache = !cachedTokens_.empty() &&
                    reusedPromptTokenCount == cachedTokens_.size();
            if (!canReuseCache) {
                // 首轮与历史裁剪、清空消息、System Prompt 变更后的回退路径，都保持
                // 原先“先清空再完整 Prefill”的行为；只有确认完整前缀一致时才保留 Cache。
                resetContext();
            }
            const size_t prefillStartIndex = canReuseCache
                    ? reusedPromptTokenCount
                    : 0;
            const auto prefillStartedAt = Clock::now();
            if (prefill(promptTokens, prefillStartIndex) != static_cast<int>(promptTokens.size())) {
                return {};
            }
            cachedTokens_ = promptTokens;
            const auto prefillFinishedAt = Clock::now();
            const auto prefillMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
                    prefillFinishedAt - prefillStartedAt
            ).count();

            if (stopRequested_.load()) {
                return {};
            }

            const llama_vocab *vocab = llama_model_get_vocab(model_);
            if (vocab == nullptr) {
                return {};
            }

            llama_sampler *sampler = createSampler(samplingConfig);
            if (sampler == nullptr) {
                return {};
            }

            std::string answer;
            // llama.cpp 的某些 Token 只包含 UTF-8 字符的一部分字节。不能把半个
            // 中文字符直接交给 JNI 的 NewStringUTF，否则会在 Kotlin 侧变成 '?'。
            std::string pendingUtf8Piece;
            int generatedTokenCount = 0;
            int64_t firstTokenMillis = -1;
            const auto decodeStartedAt = Clock::now();
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
                pendingUtf8Piece += piece;
                ++generatedTokenCount;
                if (firstTokenMillis < 0) {
                    firstTokenMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
                            Clock::now() - generationStartedAt
                    ).count();
                }

                // 只有累积成完整 UTF-8 文本后才回调 Kotlin；仍保持逐 Token 的流式体验，
                // 但不会把半个多字节字符转换成问号。
                if (!pendingUtf8Piece.empty() && isCompleteValidUtf8(pendingUtf8Piece) &&
                    !onToken(pendingUtf8Piece)) {
                    answer.clear();
                    break;
                }
                if (isCompleteValidUtf8(pendingUtf8Piece)) {
                    pendingUtf8Piece.clear();
                }

                if (stopRequested_.load()) {
                    break;
                }

                llama_sampler_accept(sampler, token);

                if (!decodeToken(token)) {
                    answer.clear();
                    break;
                }
                cachedTokens_.push_back(token);
            }

            llama_sampler_free(sampler);
            const auto generationFinishedAt = Clock::now();
            const auto decodeMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
                    generationFinishedAt - decodeStartedAt
            ).count();
            const auto totalMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
                    generationFinishedAt - generationStartedAt
            ).count();
            onMetrics(
                    generatedTokenCount,
                    static_cast<int>(prefillStartIndex),
                    prefillMillis,
                    firstTokenMillis < 0 ? totalMillis : firstTokenMillis,
                    decodeMillis,
                    totalMillis
            );
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

        llama_sampler *createSampler(const SamplingConfig &config) const {
            // temperature=0 保持原先的 greedy 语义：每一步总选概率最高的 token。
            if (config.temperature <= 0.0f) {
                return llama_sampler_init_greedy();
            }

            auto chainParams = llama_sampler_chain_default_params();
            llama_sampler *chain = llama_sampler_chain_init(chainParams);
            if (chain == nullptr) {
                return nullptr;
            }

            // 每个 sampler 的所有权转交给 chain；最后的 dist 负责按 seed 抽样。
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(config.topK));
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(config.topP, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(config.minP, 1));
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                    64,
                    config.repeatPenalty,
                    0.0f,
                    0.0f
            ));
            llama_sampler_chain_add(chain, llama_sampler_init_temp(config.temperature));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(config.seed));
            return chain;
        }

    private:
        static bool isCompleteValidUtf8(const std::string &text) {
            for (size_t index = 0; index < text.size();) {
                const auto first = static_cast<unsigned char>(text[index]);
                size_t sequenceLength = 0;
                if ((first & 0x80) == 0) {
                    sequenceLength = 1;
                } else if ((first & 0xE0) == 0xC0) {
                    sequenceLength = 2;
                } else if ((first & 0xF0) == 0xE0) {
                    sequenceLength = 3;
                } else if ((first & 0xF8) == 0xF0) {
                    sequenceLength = 4;
                } else {
                    return false;
                }
                if (index + sequenceLength > text.size()) {
                    return false;
                }
                for (size_t offset = 1; offset < sequenceLength; ++offset) {
                    if ((static_cast<unsigned char>(text[index + offset]) & 0xC0) != 0x80) {
                        return false;
                    }
                }
                index += sequenceLength;
            }
            return true;
        }

        static size_t commonPrefixLength(
                const std::vector<llama_token> &first,
                const std::vector<llama_token> &second) {
            size_t index = 0;
            while (index < first.size() && index < second.size() && first[index] == second[index]) {
                ++index;
            }
            return index;
        }

        void clearCachedTokens() {
            cachedTokens_.clear();
            nextPosition_ = 0;
        }

        llama_model *model_ = nullptr;
        llama_context *context_ = nullptr;
        llama_pos nextPosition_ = 0;
        // Kotlin 消息是事实来源；该前缀仅用于判断 Context 中哪些 Token 可安全复用。
        std::vector<llama_token> cachedTokens_;
        std::atomic_bool stopRequested_ = false;
    };
}

NativeRuntime *toRuntime(jlong handle) {
    return reinterpret_cast<NativeRuntime *>(handle);
}

// Kotlin 的 List<ChatMessage> 会以两个并行 String 数组传进 JNI。两个入口（计数、
// 生成）共用此转换，确保它们看到的是同一组 role/content。
bool readChatHistory(
        JNIEnv *env,
        jobjectArray roles,
        jobjectArray contents,
        std::vector<ChatMessage> *history) {
    if (roles == nullptr || contents == nullptr || history == nullptr) {
        return false;
    }

    const jsize messageCount = env->GetArrayLength(roles);
    if (messageCount == 0 || messageCount != env->GetArrayLength(contents)) {
        return false;
    }

    history->clear();
    history->reserve(static_cast<size_t>(messageCount));
    for (jsize index = 0; index < messageCount; ++index) {
        auto *role = static_cast<jstring>(env->GetObjectArrayElement(roles, index));
        auto *content = static_cast<jstring>(env->GetObjectArrayElement(contents, index));
        if (role == nullptr || content == nullptr) {
            if (role != nullptr) {
                env->DeleteLocalRef(role);
            }
            if (content != nullptr) {
                env->DeleteLocalRef(content);
            }
            return false;
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
            env->DeleteLocalRef(role);
            env->DeleteLocalRef(content);
            return false;
        }

        history->push_back({roleChars, contentChars});
        env->ReleaseStringUTFChars(role, roleChars);
        env->ReleaseStringUTFChars(content, contentChars);
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
    }
    return true;
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
JNIEXPORT jstring JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeModelDescription(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr) {
        return env->NewStringUTF("");
    }
    return env->NewStringUTF(runtime->modelDescription().c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeModelMetadata(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jstring key) {
    auto *runtime = toRuntime(handle);
    const char *keyChars = key == nullptr ? nullptr : env->GetStringUTFChars(key, nullptr);
    if (runtime == nullptr || keyChars == nullptr) {
        if (keyChars != nullptr) {
            env->ReleaseStringUTFChars(key, keyChars);
        }
        return env->NewStringUTF("");
    }
    const std::string value = runtime->modelMetadata(keyChars);
    env->ReleaseStringUTFChars(key, keyChars);
    return env->NewStringUTF(value.c_str());
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
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeResetContext(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle) {
    auto *runtime = toRuntime(handle);
    if (runtime != nullptr) {
        runtime->resetContext();
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeSetThreads(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jint generationThreads,
        jint batchThreads) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr) {
        return 0;
    }

    return static_cast<jint>(runtime->setThreads(generationThreads, batchThreads));
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
Java_com_arick_androidlocalllmlab_NativeLlmBridge_nativeCountChatPromptTokens(
        JNIEnv *env,
        jobject /* thiz */,
        jlong handle,
        jobjectArray roles,
        jobjectArray contents) {
    auto *runtime = toRuntime(handle);
    std::vector<ChatMessage> history;
    if (runtime == nullptr || !readChatHistory(env, roles, contents, &history)) {
        return 0;
    }

    return runtime->countChatPromptTokens(history);
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

    // 独立 Prefill 实验从空 Context 开始，不能混用聊天的增量 KV Cache。
    runtime->resetContext();
    return runtime->prefill(tokens, 0);
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
        jfloat temperature,
        jint topK,
        jfloat topP,
        jfloat minP,
        jfloat repeatPenalty,
        jint seed,
        jobject generationCallback) {
    auto *runtime = toRuntime(handle);
    if (runtime == nullptr || maxTokens <= 0 || generationCallback == nullptr) {
        return env->NewStringUTF("");
    }

    std::vector<ChatMessage> history;
    if (!readChatHistory(env, roles, contents, &history)) {
        return env->NewStringUTF("");
    }

    jclass callbackClass = env->GetObjectClass(generationCallback);
    jmethodID onPromptMethod = callbackClass == nullptr
            ? nullptr
            : env->GetMethodID(callbackClass, "onPrompt", "(Ljava/lang/String;I)V");
    jmethodID onTokenMethod = callbackClass == nullptr
            ? nullptr
            : env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
    jmethodID onMetricsMethod = callbackClass == nullptr
            ? nullptr
            : env->GetMethodID(callbackClass, "onMetrics", "(IIJJJJ)V");
    if (onPromptMethod == nullptr || onTokenMethod == nullptr || onMetricsMethod == nullptr) {
        env->ExceptionClear();
        return env->NewStringUTF("");
    }

    const std::string answer = runtime->generate(
            history,
            maxTokens,
            SamplingConfig{
                    temperature,
                    topK,
                    topP,
                    minP,
                    repeatPenalty,
                    static_cast<uint32_t>(seed)
            },
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
            },
            [env, generationCallback, onMetricsMethod](
                    int generatedTokenCount,
                    int reusedPromptTokenCount,
                    int64_t prefillMillis,
                    int64_t firstTokenMillis,
                    int64_t decodeMillis,
                    int64_t totalMillis) {
                env->CallVoidMethod(
                        generationCallback,
                        onMetricsMethod,
                        static_cast<jint>(generatedTokenCount),
                        static_cast<jint>(reusedPromptTokenCount),
                        static_cast<jlong>(prefillMillis),
                        static_cast<jlong>(firstTokenMillis),
                        static_cast<jlong>(decodeMillis),
                        static_cast<jlong>(totalMillis)
                );
                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                }
            }
    );
    return env->NewStringUTF(answer.c_str());
}
