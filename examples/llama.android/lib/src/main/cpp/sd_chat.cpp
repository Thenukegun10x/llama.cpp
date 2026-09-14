#include <android/log.h>
#include <jni.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "stable-diffusion.h"

static sd_ctx_t * g_sd_ctx = nullptr;
static std::mutex g_sd_mutex;

static void sd_android_log_callback(enum sd_log_level_t level, const char * text, void * /*data*/) {
    int android_level;
    switch (level) {
        case SD_LOG_DEBUG: android_level = ANDROID_LOG_DEBUG; break;
        case SD_LOG_INFO:  android_level = ANDROID_LOG_INFO;  break;
        case SD_LOG_WARN:  android_level = ANDROID_LOG_WARN;  break;
        case SD_LOG_ERROR: android_level = ANDROID_LOG_ERROR; break;
        default:           android_level = ANDROID_LOG_VERBOSE;
    }
    __android_log_print(android_level, "sd-chat", "%s", text);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_ImageGenEngineImpl_callNativeInit(
    JNIEnv * /*env*/, jobject /*unused*/) {
    sd_set_log_callback(sd_android_log_callback, nullptr);
    __android_log_print(ANDROID_LOG_INFO, "sd-chat", "SD backend initialized");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_ImageGenEngineImpl_callNativeSystemInfo(
    JNIEnv * env, jobject /*unused*/) {
    return env->NewStringUTF(sd_get_system_info());
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_arm_aichat_internal_ImageGenEngineImpl_callNativeLoadModel(
    JNIEnv * env, jobject /*unused*/, jstring jmodel_path) {
    std::lock_guard<std::mutex> lock(g_sd_mutex);

    if (g_sd_ctx) {
        free_sd_ctx(g_sd_ctx);
        g_sd_ctx = nullptr;
    }

    const auto * model_path = env->GetStringUTFChars(jmodel_path, 0);
    __android_log_print(ANDROID_LOG_INFO, "sd-chat", "Loading model: %s", model_path);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path;
    ctx_params.enable_mmap = true;
    ctx_params.n_threads = std::max(2, std::min(4, (int) sysconf(_SC_NPROCESSORS_ONLN) - 2));

    g_sd_ctx = new_sd_ctx(&ctx_params);
    env->ReleaseStringUTFChars(jmodel_path, model_path);

    if (!g_sd_ctx) {
        __android_log_print(ANDROID_LOG_ERROR, "sd-chat", "Failed to create SD context");
        return JNI_FALSE;
    }
    __android_log_print(ANDROID_LOG_INFO, "sd-chat", "Model loaded");
    return JNI_TRUE;
}

static bool save_png(const char * path, const uint8_t * pixels, int width, int height, int channels) {
    FILE * fp = fopen(path, "wb");
    if (!fp) return false;

    auto write_be32 = [&](uint32_t v) {
        uint8_t b[4] = { (uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v };
        fwrite(b, 1, 4, fp);
    };
    auto write_u32_crc = [&](const uint8_t * data, size_t len) {
        static const uint32_t crc_table[256] = {
            0x00000000,0x77073096,0xEE0E612C,0x990951BA,0x076DC419,0x706AF48F,0xE963A535,0x9E6495A3,
            0x0EDB8832,0x79DCB8A4,0xE0D5E91E,0x97D2D988,0x09B64C2B,0x7EB17CBD,0xE7B82D07,0x90BF1D91,
            0x1DB71064,0x6AB020F2,0xF3B97148,0x84BE41DE,0x1ADAD47D,0x6DDDE4EB,0xF4D4B551,0x83D385C7,
            0x136C9856,0x646BA8C0,0xFD62F97A,0x8A65C9EC,0x14015C4F,0x63066CD9,0xFA0F3D63,0x8D080DF5,
            0x3B6E20C8,0x4C69105E,0xD56041E4,0xA2677172,0x3C03E4D1,0x4B04D447,0xD20D85FD,0xA50AB56B,
            0x35B5A8FA,0x42B2986C,0xDBBBC9D6,0xACBCF940,0x32D86CE3,0x45DF5C75,0xDCD60DCF,0xABD13D59,
            0x26D930AC,0x51DE003A,0xC8D75180,0xBFD06116,0x21B4F4B5,0x56B3C423,0xCFBA9599,0xB8BDA50F,
            0x2802B89E,0x5F058808,0xC60CD9B2,0xB10BE924,0x2F6F7C87,0x58684C11,0xC1611DAB,0xB6662D3D,
            0x76DC4190,0x01DB7106,0x98D220BC,0xEFD5102A,0x71B18589,0x06B6B51F,0x9FBFE4A5,0xE8B8D433,
            0x7807C9A2,0x0F00F934,0x9609A88E,0xE10E9818,0x7F6A0DBB,0x086D3D2D,0x91646C97,0xE6635C01,
            0x6B6B51F4,0x1C6C6162,0x856530D8,0xF262004E,0x6C0695ED,0x1B01A57B,0x8208F4C1,0xF50FC457,
            0x65B0D9C6,0x12B7E950,0x8BBEB8EA,0xFCB9887C,0x62DD1DDF,0x15DA2D49,0x8CD37CF3,0xFBD44C65,
            0x4DB26158,0x3AB551CE,0xA3BC0074,0xD4BB30E2,0x4A4FA541,0x3D3D95D7,0xA4365E6D,0xD35D6EFB,
            0x43D99E6A,0x34DE86FC,0xAD541746,0xDA537FD0,0x447CE773,0x333BD773,0xAA3266C9,0xDD3D455F,
            0x50F82B61,0x2781E3E3,0xBE232659,0xC9240DD0,0x57B3B873,0x20B4A0B6,0xB9BDF30C,0xCEBAE37A,
            0x5E0F2EEB,0x2907BE75,0xB00FEFAF,0xC708FE31,0x59C941D2,0x2ECF4144,0xB7CD10FE,0xC0CAB468,
            0xDB2264AD,0xAC25393B,0x352CB680,0x422C86E6,0xDC23F3E5,0xAB24C373,0x322D92C9,0x452AA25F,
            0xD5BEBF8E,0xA2B98F18,0x3BB0DEA2,0x4CB7B034,0xD2F325D7,0xA5F41541,0x3CFB44FB,0x4BFC556D,
            0xC6A6A483,0xB1A1D4F7,0x28A8FB4D,0x5FAFEBDB,0xC1E1BEE8,0xB6E68E7E,0x2FB8FFC4,0x58C0FF52,
            0xC84F33C3,0xBF3B0337,0x26300B8D,0x5137311B,0xCF69A2F8,0xB86E926E,0x2167C3D4,0x5661B342,
            0xFCB2EDDA,0x8BBFA57C,0x12E823C6,0x65EE1350,0xFBE59A2A,0x8EE3AABC,0x17EAFB06,0x60ECCB70,
            0x90A8D7E1,0xE7A89977,0x7EA1BACD,0x09A74A5B,0x97E6C478,0xE0E1F43E,0x79E8A584,0x0ECFA512,
            0xC3C24A3D,0xB4C53A6B,0x2DCC7FD1,0x5ACB4F47,0xC4F5DA74,0xB3F2EA02,0x2AA3BBB8,0x5DA48B2E,
            0xCDB3E62F,0xBA86D759,0x238F8427,0x54A8B4B1,0xCAE63182,0xBDD20114,0x242D50AE,0x531A6038,
        };
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < len; i++) {
            crc = crc_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
        write_be32(crc ^ 0xFFFFFFFF);
    };

    int bytes_per_pixel = (channels == 4) ? 4 : 3;
    int color_type = (channels == 4) ? 6 : 2;
    size_t row_bytes = width * bytes_per_pixel + 1;

    // PNG signature
    uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, fp);

    // IHDR
    {
        uint8_t ihdr_data[13];
        auto * p = ihdr_data;
        auto write_be32_to = [&](uint32_t v) {
            p[0]=(uint8_t)(v>>24); p[1]=(uint8_t)(v>>16);
            p[2]=(uint8_t)(v>>8);  p[3]=(uint8_t)v; p+=4;
        };
        write_be32_to(width);
        write_be32_to(height);
        *p++ = 8;
        *p++ = (uint8_t)color_type;
        *p++ = 0; *p++ = 0; *p++ = 0;

        uint8_t chunk_type[4] = {'I','H','D','R'};
        write_be32(13);
        fwrite(chunk_type, 1, 4, fp);
        fwrite(ihdr_data, 1, 13, fp);
        uint8_t crc_in[17];
        memcpy(crc_in, chunk_type, 4);
        memcpy(crc_in + 4, ihdr_data, 13);
        write_u32_crc(crc_in, 17);
    }

    // IDAT — uncompressed deflate blocks, written in two passes to avoid large buffer
    size_t raw_size = height * row_bytes;
    size_t num_blocks = (raw_size + 65534) / 65535;
    size_t idat_size = 2 + num_blocks * 5 + raw_size + 4; // zlib header + blocks + adler32

    // Zlib header (CMF=0x78, FLG=0x01)
    uint8_t chunk_type[4] = {'I','D','A','T'};
    write_be32((uint32_t)idat_size);
    fwrite(chunk_type, 1, 4, fp);

    // Zlib header
    uint8_t zlib[2] = {0x78, 0x01};
    fwrite(zlib, 1, 2, fp);

    // Deflate blocks
    for (size_t offset = 0; offset < raw_size; ) {
        size_t block_len = (raw_size - offset > 65535) ? 65535 : (raw_size - offset);
        bool is_final = (offset + block_len >= raw_size);
        uint8_t hdr[5];
        hdr[0] = (uint8_t)(is_final ? 1 : 0);
        hdr[1] = (uint8_t)(block_len & 0xFF);
        hdr[2] = (uint8_t)((block_len >> 8) & 0xFF);
        uint16_t nlen = (uint16_t)(~block_len & 0xFFFF);
        hdr[3] = (uint8_t)(nlen & 0xFF);
        hdr[4] = (uint8_t)((nlen >> 8) & 0xFF);
        fwrite(hdr, 1, 5, fp);

        for (size_t i = 0; i < block_len; i += row_bytes) {
            size_t row = (offset + i) / row_bytes;
            fputc(0, fp); // filter none
            fwrite(pixels + row * width * bytes_per_pixel, 1, width * bytes_per_pixel, fp);
        }
        offset += block_len;
    }

    // Adler32 over raw filtered scanlines
    uint32_t s1 = 1, s2 = 0;
    const uint32_t MOD_ADLER = 65521;
    for (int row = 0; row < height; row++) {
        s1 = (s1 + 0) % MOD_ADLER;
        s2 = (s2 + s1) % MOD_ADLER;
        for (int col = 0; col < width * bytes_per_pixel; col++) {
            uint8_t byte_val = pixels[row * width * bytes_per_pixel + col];
            s1 = (s1 + byte_val) % MOD_ADLER;
            s2 = (s2 + s1) % MOD_ADLER;
        }
    }
    uint8_t adler[4] = {
        (uint8_t)((s2 >> 8) & 0xFF), (uint8_t)(s2 & 0xFF),
        (uint8_t)((s1 >> 8) & 0xFF), (uint8_t)(s1 & 0xFF)
    };
    fwrite(adler, 1, 4, fp);

    // IDAT CRC
    {
        std::vector<uint8_t> crc_in;
        crc_in.reserve(4 + idat_size);
        crc_in.insert(crc_in.end(), chunk_type, chunk_type + 4);
        write_u32_crc(crc_in.data(), crc_in.size());
    }

    // IEND
    {
        uint8_t iend[4] = {'I','E','N','D'};
        write_be32(0);
        fwrite(iend, 1, 4, fp);
        write_u32_crc(iend, 4);
    }

    fclose(fp);
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_ImageGenEngineImpl_callNativeGenerateImage(
        JNIEnv * env,
        jobject /*unused*/,
        jstring jprompt,
        jstring jnegative_prompt,
        jint width,
        jint height,
        jint steps,
        jfloat cfg_scale,
        jlong seed,
        jstring joutput_path
) {
    std::lock_guard<std::mutex> lock(g_sd_mutex);

    if (!g_sd_ctx) {
        __android_log_print(ANDROID_LOG_ERROR, "sd-chat", "Model not loaded");
        return nullptr;
    }

    const auto * prompt = env->GetStringUTFChars(jprompt, 0);
    const auto * negative_prompt = env->GetStringUTFChars(jnegative_prompt, 0);
    const auto * output_path = env->GetStringUTFChars(joutput_path, 0);

    sd_img_gen_params_t img_params;
    sd_img_gen_params_init(&img_params);

    img_params.prompt = prompt;
    img_params.negative_prompt = negative_prompt;
    img_params.width = width;
    img_params.height = height;
    img_params.sample_params.sample_steps = steps;
    img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    img_params.sample_params.guidance.txt_cfg = cfg_scale;
    img_params.seed = seed;
    img_params.batch_count = 1;

    __android_log_print(ANDROID_LOG_INFO, "sd-chat",
        "Generating: %dx%d, steps=%d, cfg=%.1f, seed=%ld",
        width, height, steps, cfg_scale, (long)seed);

    sd_image_t * images = nullptr;
    int num_images = 0;

    bool ok = generate_image(g_sd_ctx, &img_params, &images, &num_images);

    env->ReleaseStringUTFChars(jprompt, prompt);
    env->ReleaseStringUTFChars(jnegative_prompt, negative_prompt);

    if (!ok || !images || num_images < 1 || !images[0].data) {
        env->ReleaseStringUTFChars(joutput_path, output_path);
        if (images) free_sd_images(images, num_images);
        __android_log_print(ANDROID_LOG_ERROR, "sd-chat", "Image generation failed");
        return nullptr;
    }

    sd_image_t & img = images[0];
    bool saved = save_png(output_path, img.data, img.width, img.height, img.channel);

    free_sd_images(images, num_images);
    env->ReleaseStringUTFChars(joutput_path, output_path);

    return saved ? env->NewStringUTF(output_path) : nullptr;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_ImageGenEngineImpl_callNativeCancel(
    JNIEnv * /*env*/, jobject /*unused*/) {
    std::lock_guard<std::mutex> lock(g_sd_mutex);
    if (g_sd_ctx) {
        sd_cancel_generation(g_sd_ctx, SD_CANCEL_ALL);
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_ImageGenEngineImpl_callNativeUnload(
    JNIEnv * /*env*/, jobject /*unused*/) {
    std::lock_guard<std::mutex> lock(g_sd_mutex);
    if (g_sd_ctx) {
        free_sd_ctx(g_sd_ctx);
        g_sd_ctx = nullptr;
        __android_log_print(ANDROID_LOG_INFO, "sd-chat", "Model unloaded");
    }
}
