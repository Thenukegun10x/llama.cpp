#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <unistd.h>
#include <sampling.h>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

/**
 * LLama resources: context, model, batch and sampler
 */
constexpr int   N_THREADS_MIN           = 2;
constexpr int   N_THREADS_MAX           = 4;
constexpr int   N_THREADS_HEADROOM      = 2;

constexpr int   DEFAULT_CONTEXT_SIZE    = 4096;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 512;
constexpr float DEFAULT_SAMPLER_TEMP    = 0.3f;

// Runtime-tunable config, set from the app via configure()/updateSampling().
// Context size and KV types apply on the next model load; sampling applies live.
static int            g_n_ctx           = DEFAULT_CONTEXT_SIZE;
static int            g_n_threads       = 0; // 0 = auto
static enum ggml_type g_type_k          = GGML_TYPE_F16;
static enum ggml_type g_type_v          = GGML_TYPE_F16;
static float          g_temp            = DEFAULT_SAMPLER_TEMP;
static int            g_top_k           = 40;
static float          g_top_p           = 0.95f;
static float          g_penalty_repeat  = 1.0f;

static llama_model                      * g_model;
static llama_context                    * g_context;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler;

using json = common_json;

static std::vector<common_chat_tool>      g_tools;
static common_chat_params                 g_active_chat_params;
static common_chat_parser_params          g_active_parser_params;
static bool                               g_has_active_parser = false;

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    // Set llama log handler to Android
    llama_log_set(aichat_android_log_callback, nullptr);

    // Loading all CPU backend variants
    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    LOGi("Loading backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    // Initialize backends
    llama_backend_init();
    LOGi("Backend initiated; Log handler set.");
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_load(JNIEnv *env, jobject, jstring jmodel_path) {
    llama_model_params model_params = llama_model_default_params();
    model_params.use_extra_bufts = false;

    // Keep the large output projection (vocab size 128k, ~200MB) on CPU to avoid
    // mobile GPU storage buffer range limitations (>128MB) and driver fence hangs,
    // while keeping all 29 repeating transformer layers accelerated on Vulkan GPU.
    static const llama_model_tensor_buft_override tensor_buft_overrides[] = {
        { "output\\.weight", ggml_backend_cpu_buffer_type() },
        { "^output$",        ggml_backend_cpu_buffer_type() },
        { nullptr,           nullptr }
    };
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    LOGd("%s: Loading model from: \n%s\n", __func__, model_path);

    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        return 1;
    }
    g_model = model;
    return 0;
}

static llama_context *init_context(llama_model *model, const int n_ctx = 0) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    const int ctx = n_ctx > 0 ? n_ctx : g_n_ctx;

    // Multi-threading setup; explicit user value wins, otherwise a capped default
    const int n_threads = g_n_threads > 0
        ? g_n_threads
        : std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
                                           (int) sysconf(_SC_NPROCESSORS_ONLN) -
                                           N_THREADS_HEADROOM));
    LOGi("%s: Using %d threads", __func__, n_threads);

    // Context parameters setup
    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (ctx > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
             __func__, trained_context_size, ctx);
    }
    ctx_params.n_ctx = ctx;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.type_k = g_type_k;
    ctx_params.type_v = g_type_v;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static common_sampler *new_sampler() {
    common_params_sampling sparams;
    sparams.temp           = g_temp;
    sparams.top_k          = g_top_k;
    sparams.top_p          = g_top_p;
    sparams.penalty_repeat = g_penalty_repeat;
    return common_sampler_init(g_model, sparams);
}

// Applies context size + KV cache type + thread count on the next model load.
// kv_type: 0 = F16, 1 = Q8_0, 2 = Q4_0. n_threads <= 0 = auto.
extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeConfigure(JNIEnv * /*env*/, jobject /*unused*/,
        jint n_ctx, jint kv_type, jint n_threads) {
    g_n_ctx = n_ctx > 0 ? n_ctx : DEFAULT_CONTEXT_SIZE;
    g_n_threads = n_threads > 0 ? n_threads : 0;
    enum ggml_type kv = GGML_TYPE_F16;
    if (kv_type == 1) {
        kv = GGML_TYPE_Q8_0;
    } else if (kv_type == 2) {
        kv = GGML_TYPE_Q4_0;
    }
    g_type_k = kv;
    g_type_v = kv;
    LOGi("%s: n_ctx=%d kv=%d n_threads=%d", __func__, g_n_ctx, (int) kv_type, g_n_threads);
}

// Updates sampling live; takes effect on the next prompt when idle.
extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeUpdateSampling(JNIEnv * /*env*/, jobject /*unused*/,
        jfloat temp, jint top_k, jfloat top_p, jfloat penalty_repeat) {
    g_temp           = temp;
    g_top_k          = top_k;
    g_top_p          = top_p;
    g_penalty_repeat = penalty_repeat;
    if (g_model != nullptr && g_sampler != nullptr) {
        common_sampler_free(g_sampler);
        g_sampler = new_sampler();
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_prepare(JNIEnv * /*env*/, jobject /*unused*/) {
    auto *context = init_context(g_model);
    if (!context) { return 1; }
    g_context = context;
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    g_sampler = new_sampler();
    return 0;
}

static std::string get_backend() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.push_back(ggml_backend_reg_name(reg));
        }
    }
    return backends.empty() ? "CPU" : join(backends, ",");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_benchModel(JNIEnv *env, jobject /*unused*/, jint pp, jint tg,
                                                      jint pl, jint nr) {
    auto *context = init_context(g_model, pp);
    if (!context) {
        const auto *const err_msg = "Fail to init_context! Bench aborted.";
        LOGe(err_msg);
        return env->NewStringUTF(err_msg);
    }

    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(context);
    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp = %d)", pp);

        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 0, i, {0}, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, g_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, {j}, true);
            }

            if (llama_decode(context, g_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    llama_free(context);

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;

    const auto backend = get_backend();
    std::stringstream result;
    result << std::setprecision(3);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | "
           << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
    return env->NewStringUTF(result.str().c_str());
}


/**
 * Completion loop's long-term states:
 * - chat management
 * - position tracking
 */
constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";
constexpr const char *ROLE_TOOL         = "tool";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;
    g_has_active_parser = false;

    if (clear_kv_cache)
        llama_memory_clear(llama_get_memory(g_context), false);
}

/**
 * TODO-hyin: implement sliding-window version as a better alternative
 *
 * Context shifting by discarding the older half of the tokens appended after system prompt:
 * - take the [system_prompt_position] first tokens from the original prompt
 * - take half of the last (system_prompt_position - system_prompt_position) tokens
 * - recompute the logits in batches
 */
static void shift_context() {
    const int n_discard = (current_position - system_prompt_position) / 2;
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static std::string chat_format_single_with_tools(const struct common_chat_templates * tmpls,
                                                const std::vector<common_chat_msg> & past_msg,
                                                const common_chat_msg & new_msg,
                                                bool add_ass,
                                                bool use_jinja) {
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos = llama_vocab_get_add_bos(vocab);
    inputs.add_eos = llama_vocab_get_add_eos(vocab);
    if (!g_tools.empty()) {
        inputs.tools = g_tools;
    }
    inputs.enable_thinking = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    }
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;

    auto params = common_chat_templates_apply(tmpls, inputs);
    auto fmt_new_msg = params.prompt;

    if (add_ass) {
        g_active_chat_params = params;
        g_active_parser_params = common_chat_parser_params(g_active_chat_params);
        g_active_parser_params.parse_tool_calls = true;
        g_active_parser_params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        if (!g_active_chat_params.parser.empty()) {
            common_peg_arena arena;
            arena.load(g_active_chat_params.parser);
            g_active_parser_params.parser = std::move(arena);
            g_has_active_parser = true;
            LOGi("%s: Active chat parser initialized (format=%s, thinking=%d)",
                 __func__, common_chat_format_name(g_active_chat_params.format),
                 (int) g_active_chat_params.supports_thinking);
        } else {
            g_has_active_parser = false;
        }
    }

    if (fmt_new_msg.size() >= fmt_past_msg.size()) {
        ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    } else {
        ss << fmt_new_msg;
    }
    return ss.str();
}

static std::string chat_add_and_format(const std::string &role, const std::string &content,
                                      const std::string &tool_name = "", const std::string &tool_call_id = "") {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    new_msg.tool_name = tool_name;
    new_msg.tool_call_id = tool_call_id;

    std::string formatted;
    bool add_ass = (role == ROLE_USER || role == ROLE_TOOL);
    try {
        formatted = chat_format_single_with_tools(
                g_chat_templates.get(), chat_msgs, new_msg, add_ass, /* use_jinja */ true);
    } catch (const std::exception &e) {
        LOGw("%s: Jinja format with tools failed: %s, falling back...", __func__, e.what());
        try {
            if (role == ROLE_TOOL) {
                new_msg.role = ROLE_USER;
            }
            formatted = common_chat_format_single(
                    g_chat_templates.get(), chat_msgs, new_msg, add_ass, /* use_jinja */ true);
        } catch (const std::exception &e2) {
            LOGw("%s: Jinja format without tools also failed: %s, falling back to legacy...", __func__, e2.what());
            try {
                formatted = common_chat_format_single(
                        g_chat_templates.get(), chat_msgs, new_msg, add_ass, /* use_jinja */ false);
            } catch (const std::exception &e3) {
                LOGe("%s: Legacy format also failed: %s, falling back to raw formatting", __func__, e3.what());
                if (role == ROLE_SYSTEM) {
                    formatted = "<|system|>\n" + content + "\n";
                } else if (role == ROLE_USER || role == ROLE_TOOL) {
                    formatted = "<|user|>\n" + content + "\n<|assistant|>\n";
                } else {
                    formatted = content;
                }
            }
        }
    }
    chat_msgs.push_back(new_msg);
    LOGi("%s: Formatted and added %s message: \n%s\n", __func__, role.c_str(), formatted.c_str());
    return formatted;
}

static void record_assistant_message(const std::string & response) {
    common_chat_msg new_msg;
    new_msg.role = ROLE_ASSISTANT;
    new_msg.content = response;
    if (g_has_active_parser) {
        try {
            common_chat_msg parsed = common_chat_parse(response, false, g_active_parser_params);
            if (!parsed.tool_calls.empty()) {
                new_msg.tool_calls = parsed.tool_calls;
                new_msg.reasoning_content = parsed.reasoning_content;
                new_msg.content = parsed.content;
                LOGi("%s: Recorded assistant tool call in chat_msgs (name=%s, count=%zu)",
                     __func__, parsed.tool_calls[0].name.c_str(), parsed.tool_calls.size());
            }
        } catch (const std::exception & e) {
            LOGw("%s: Failed to parse assistant response: %s", __func__, e.what());
        }
    }
    chat_msgs.push_back(new_msg);
}

/**
 * Completion loop's short-term states:
 * - stop generation position
 * - token chars caching
 * - current assistant message being generated
 */
static llama_pos stop_generation_position;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;
static bool g_prompt_ends_with_think = false;

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
    g_prompt_ends_with_think = false;
}

// Returns int[2] = {tokens currently in context, context size}.
extern "C"
JNIEXPORT jintArray JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeContextUsage(JNIEnv *env, jobject /*unused*/) {
    const jint vals[2] = {(jint) current_position, (jint) g_n_ctx};
    jintArray out = env->NewIntArray(2);
    if (out != nullptr) {
        env->SetIntArrayRegion(out, 0, 2, vals);
    }
    return out;
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    LOGi("%s: Decode %d tokens starting at position %d (compute_last_logit=%d)",
         __func__, (int) tokens.size(), start_pos, (int) compute_last_logit);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(batch);
        LOGi("%s: Preparing batch of %d tokens (offset %d)", __func__, cur_batch_size, i);

        // Shift context if current batch cannot fit into the context
        if (start_pos + i + cur_batch_size >= g_n_ctx - OVERFLOW_HEADROOM) {
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        // Add tokens to the batch with proper positions
        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        // Decode this batch
        LOGi("%s: Calling llama_decode(batch_size=%d)...", __func__, cur_batch_size);
        const int decode_result = llama_decode(context, batch);
        LOGi("%s: llama_decode returned %d", __func__, decode_result);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    LOGi("%s: All batches decoded successfully", __func__);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Obtain system prompt from JEnv
    const auto *system_prompt = env->GetStringUTFChars(jsystem_prompt, nullptr);
    LOGd("%s: System prompt received: \n%s", __func__, system_prompt);
    std::string formatted_system_prompt(system_prompt);

    // Format system prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, system_prompt);
    }
    env->ReleaseStringUTFChars(jsystem_prompt, system_prompt);

    // Tokenize system prompt
    const auto system_tokens = common_tokenize(g_context, formatted_system_prompt,
                                               has_chat_template, has_chat_template);
    for (auto id: system_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Handle context overflow
    const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
    if ((int) system_tokens.size() > max_batch_size) {
        LOGe("%s: System prompt too long for context! %d tokens, max: %d",
             __func__, (int) system_tokens.size(), max_batch_size);
        return 1;
    }

    // Decode system tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    system_prompt_position = current_position = (int) system_tokens.size();
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    // Reset short-term states
    reset_short_term_states();

    // Obtain and tokenize user prompt
    const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
    LOGd("%s: User prompt received: \n%s", __func__, user_prompt);
    std::string formatted_user_prompt(user_prompt);

    // Format user prompt if applicable
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_user_prompt = chat_add_and_format(ROLE_USER, user_prompt);
    }
    env->ReleaseStringUTFChars(juser_prompt, user_prompt);

    // If chat template prefilled <think> at the end of the prompt, the model will generate
    // the thought body directly without emitting <think>. Flag this to emit <think>\n to stream.
    {
        std::string trimmed = formatted_user_prompt;
        while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\n' || trimmed.back() == '\r' || trimmed.back() == '\t')) {
            trimmed.pop_back();
        }
        if (trimmed.size() >= 7 && trimmed.substr(trimmed.size() - 7) == "<think>") {
            g_prompt_ends_with_think = true;
            LOGi("%s: Chat template prefilled <think>, will emit to stream", __func__);
        }
    }

    // Decode formatted user prompts
    auto user_tokens = common_tokenize(g_context, formatted_user_prompt, has_chat_template, has_chat_template);
    for (auto id: user_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    // Ensure user prompt doesn't exceed the context size by truncating if necessary.
    const int user_prompt_size = (int) user_tokens.size();
    const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
    if (user_prompt_size > max_batch_size) {
        const int skipped_tokens = user_prompt_size - max_batch_size;
        user_tokens.resize(max_batch_size);
        LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }

    // Decode user tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update position
    current_position += user_prompt_size;
    stop_generation_position = current_position + user_prompt_size + n_predict;
    return 0;
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    if (g_prompt_ends_with_think) {
        g_prompt_ends_with_think = false;
        assistant_ss << "<think>\n";
        return env->NewStringUTF("<think>\n");
    }

    // Infinite text generation via context shifting
    if (current_position >= g_n_ctx - OVERFLOW_HEADROOM) {
        LOGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    // Stop if reaching the marked position
    if (current_position >= stop_generation_position) {
        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        record_assistant_message(assistant_ss.str());
        return nullptr;
    }

    // Sample next token
    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token_id, true);
    LOGi("%s: Sampled token %d ('%s')", __func__, new_token_id, common_token_to_piece(g_context, new_token_id).c_str());

    // Stop if next token is EOG
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id)) {
        LOGi("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        record_assistant_message(assistant_ss.str());
        return nullptr;
    }

    // Populate the batch with new token, then decode
    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("%s: llama_decode() failed for generated token %d", __func__, new_token_id);
        return nullptr;
    }

    // Update position
    current_position++;

    // If not EOG, convert to text
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    // Create and return a valid UTF-8 Java string
    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unload(JNIEnv * /*unused*/, jobject /*unused*/) {
    // Reset long-term & short-term states
    reset_long_term_states();
    reset_short_term_states();

    // Free up resources
    g_tools.clear();
    g_has_active_parser = false;
    common_sampler_free(g_sampler);
    g_sampler = nullptr;
    g_chat_templates.reset();
    llama_batch_free(g_batch);
    llama_free(g_context);
    g_context = nullptr;
    llama_model_free(g_model);
    g_model = nullptr;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_shutdown(JNIEnv *, jobject /*unused*/) {
    llama_backend_free();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeSetTools(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jtools_json
) {
    g_tools.clear();
    g_has_active_parser = false;
    if (!jtools_json) return;

    const char *tools_str = env->GetStringUTFChars(jtools_json, nullptr);
    if (tools_str && tools_str[0] != '\0') {
        try {
            auto j = json::parse(tools_str);
            if (j.is_array()) {
                for (const auto & item : j) {
                    common_chat_tool tool;
                    if (item.contains("function")) {
                        const auto & f = item["function"];
                        tool.name = f.value("name", "");
                        tool.description = f.value("description", "");
                        if (f.contains("parameters")) {
                            tool.parameters = f["parameters"].dump();
                        }
                    } else {
                        tool.name = item.value("name", "");
                        tool.description = item.value("description", "");
                        if (item.contains("parameters")) {
                            tool.parameters = item["parameters"].dump();
                        }
                    }
                    if (!tool.name.empty()) {
                        g_tools.push_back(tool);
                    }
                }
                LOGi("%s: Registered %zu tools", __func__, g_tools.size());
            }
        } catch (const std::exception & e) {
            LOGe("%s: Failed to parse tools JSON: %s", __func__, e.what());
        }
    }
    if (tools_str) {
        env->ReleaseStringUTFChars(jtools_json, tools_str);
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeHasToolSupport(
        JNIEnv * /*env*/,
        jobject /*unused*/
) {
    if (!g_chat_templates) return JNI_FALSE;
    const auto caps = common_chat_templates_get_caps(g_chat_templates.get());
    const auto it = caps.find("supports_tools");
    return (it != caps.end() && it->second) ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeParseResponse(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jraw_text,
        jboolean jis_partial
) {
    json out;
    if (!jraw_text) {
        out["has_parser"] = false;
        out["content"] = "";
        out["reasoning_content"] = "";
        out["tool_calls"] = json::array();
        return env->NewStringUTF(out.dump().c_str());
    }

    const char *raw_chars = env->GetStringUTFChars(jraw_text, nullptr);
    std::string raw_text(raw_chars ? raw_chars : "");
    if (raw_chars) {
        env->ReleaseStringUTFChars(jraw_text, raw_chars);
    }

    if (g_has_active_parser) {
        try {
            common_chat_msg parsed = common_chat_parse(raw_text, (bool) jis_partial, g_active_parser_params);
            out["has_parser"] = true;
            out["content"] = parsed.content;
            out["reasoning_content"] = parsed.reasoning_content;
            out["tool_calls"] = json::array();
            for (const auto & tc : parsed.tool_calls) {
                out["tool_calls"].push_back({
                    {"name", tc.name},
                    {"arguments", tc.arguments},
                    {"id", tc.id}
                });
            }
            return env->NewStringUTF(out.dump().c_str());
        } catch (const std::exception & e) {
            LOGw("%s: common_chat_parse failed: %s", __func__, e.what());
        }
    }

    out["has_parser"] = false;
    out["content"] = raw_text;
    out["reasoning_content"] = "";
    out["tool_calls"] = json::array();
    return env->NewStringUTF(out.dump().c_str());
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processToolResponse(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jtool_name,
        jstring jcall_id,
        jstring jcontent,
        jint n_predict
) {
    // Reset short-term states
    reset_short_term_states();

    const char *tool_name_chars = jtool_name ? env->GetStringUTFChars(jtool_name, nullptr) : nullptr;
    const char *call_id_chars = jcall_id ? env->GetStringUTFChars(jcall_id, nullptr) : nullptr;
    const char *content_chars = env->GetStringUTFChars(jcontent, nullptr);

    std::string tool_name(tool_name_chars ? tool_name_chars : "");
    std::string call_id(call_id_chars ? call_id_chars : "");
    std::string content(content_chars ? content_chars : "");

    if (tool_name_chars) env->ReleaseStringUTFChars(jtool_name, tool_name_chars);
    if (call_id_chars) env->ReleaseStringUTFChars(jcall_id, call_id_chars);
    env->ReleaseStringUTFChars(jcontent, content_chars);

    LOGd("%s: Tool response received for '%s': \n%s", __func__, tool_name.c_str(), content.c_str());

    std::string formatted_tool_prompt = content;
    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_tool_prompt = chat_add_and_format(ROLE_TOOL, content, tool_name, call_id);
    }

    // Tokenize
    auto tool_tokens = common_tokenize(g_context, formatted_tool_prompt, has_chat_template, has_chat_template);
    for (auto id: tool_tokens) {
        LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
    }

    const int tool_prompt_size = (int) tool_tokens.size();
    const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
    if (tool_prompt_size > max_batch_size) {
        const int skipped_tokens = tool_prompt_size - max_batch_size;
        tool_tokens.resize(max_batch_size);
        LOGw("%s: Tool prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
    }

    // Decode tool tokens in batches
    if (decode_tokens_in_batches(g_context, g_batch, tool_tokens, current_position, true)) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    current_position += tool_prompt_size;
    stop_generation_position = current_position + tool_prompt_size + n_predict;
    return 0;
}
