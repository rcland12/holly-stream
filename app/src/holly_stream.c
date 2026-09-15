/*
 * Holly Stream: zero-copy object detection live stream for the Jetson Nano.
 *
 *   nvarguscamerasrc (ISP) -> nvvideoconvert (VIC scale)
 *     -> nvstreammux -> nvinfer (TensorRT YOLO) -> [nvtracker] -> nvdsosd
 *     -> nvvideoconvert (VIC) -> nvv4l2h264enc (NVENC) -> h264parse
 *     -> flvmux (+ optional ALSA/AAC audio) -> leaky queue -> rtmpsink
 *
 * Frames never leave NVMM memory, so the CPU only draws labels, runs NMS on a
 * handful of boxes and pushes bytes to the network. Everything is configured
 * through environment variables (see .env.example) and the process exits
 * non-zero on any unrecoverable error or stall so Docker restarts it.
 */

#include <glib.h>
#include <glib/gstdio.h>
#include <glib-unix.h>
#include <gst/gst.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gstnvdsmeta.h"

#define APP_DIR "/opt/holly-stream"
#define RUNTIME_DIR "/tmp/holly-stream"
#define DS_LIB_DIR "/opt/nvidia/deepstream/deepstream/lib"
#define TRTEXEC "/usr/src/tensorrt/bin/trtexec"
#define MAX_CLASSES 1024

/* ------------------------------------------------------------------------- */
/* Configuration                                                              */
/* ------------------------------------------------------------------------- */

typedef struct {
    gchar *rtmp_uri;
    gint sensor_id, capture_width, capture_height, fps;
    gint wb_mode, tnr_mode;
    gdouble tnr_strength;
    gint width, height;

    gchar *quality;
    gint bitrate, peak_bitrate, control_rate, preset_level, gop;

    gboolean audio;
    gchar *audio_device;
    gint audio_bitrate, audio_rate, audio_channels;

    gboolean detection;
    gchar *model_onnx, *model_engine, *labels_path;
    gchar *classes;
    gdouble confidence, nms_iou;
    gint interval;
    gchar *tracker;
    gint osd_mode;
    gboolean show_confidence;

    gint stats_interval, watchdog_seconds;
} Config;

static const gchar *env_str(const gchar *name, const gchar *fallback)
{
    const gchar *value = g_getenv(name);
    return (value && *value) ? value : fallback;
}

static gint env_int(const gchar *name, gint fallback)
{
    const gchar *value = g_getenv(name);
    return (value && *value) ? (gint) g_ascii_strtoll(value, NULL, 10) : fallback;
}

static gdouble env_double(const gchar *name, gdouble fallback)
{
    const gchar *value = g_getenv(name);
    return (value && *value) ? g_ascii_strtod(value, NULL) : fallback;
}

static gboolean env_bool(const gchar *name, gboolean fallback)
{
    const gchar *value = g_getenv(name);
    if (!value || !*value)
        return fallback;
    return g_ascii_strcasecmp(value, "true") == 0 || g_ascii_strcasecmp(value, "1") == 0 ||
           g_ascii_strcasecmp(value, "yes") == 0 || g_ascii_strcasecmp(value, "on") == 0;
}

static void parse_resolution(Config *c)
{
    const gchar *preset = env_str("STREAM_RESOLUTION", "720p");
    gint fps = 30;

    if (g_str_equal(preset, "1080p") || g_str_equal(preset, "1080p30")) {
        c->width = 1920; c->height = 1080;
    } else if (g_str_equal(preset, "720p") || g_str_equal(preset, "720p30")) {
        c->width = 1280; c->height = 720;
    } else if (g_str_equal(preset, "720p60")) {
        c->width = 1280; c->height = 720; fps = 60;
    } else if (g_str_equal(preset, "540p")) {
        c->width = 960; c->height = 540;
    } else if (g_str_equal(preset, "480p")) {
        c->width = 854; c->height = 480;
    } else if (g_str_equal(preset, "360p")) {
        c->width = 640; c->height = 360;
    } else if (sscanf(preset, "%dx%d", &c->width, &c->height) != 2) {
        g_printerr("[WARN] Unknown STREAM_RESOLUTION '%s', using 720p\n", preset);
        c->width = 1280; c->height = 720;
    }
    /* Encoders and the scaler want even dimensions. */
    c->width &= ~1;
    c->height &= ~1;
    c->fps = env_int("CAMERA_FPS", fps);

    /* Capture mode fed to the ISP. The hardware scaler resizes it to the output,
     * so a full-sensor mode (e.g. IMX219 3264x1848@28) keeps the widest view. */
    c->capture_width = env_int("CAPTURE_WIDTH", c->fps > 30 ? 1280 : 1920);
    c->capture_height = env_int("CAPTURE_HEIGHT", c->fps > 30 ? 720 : 1080);
}

static void parse_quality(Config *c)
{
    gint bpp = 10, peak_pct = 120;

    c->quality = g_strdup(env_str("STREAM_QUALITY", "smooth"));
    c->control_rate = 1; /* nvv4l2h264enc: 0 = VBR, 1 = CBR */
    c->preset_level = 3;
    if (g_str_equal(c->quality, "ultra")) {
        bpp = 15; peak_pct = 140; c->control_rate = 0; c->preset_level = 4;
    } else if (g_str_equal(c->quality, "high")) {
        bpp = 12; peak_pct = 130;
    } else if (g_str_equal(c->quality, "balanced")) {
        bpp = 8; peak_pct = 115; c->preset_level = 2;
    } else if (g_str_equal(c->quality, "fast")) {
        bpp = 6; peak_pct = 110; c->preset_level = 2;
    } else if (!g_str_equal(c->quality, "smooth")) {
        g_printerr("[WARN] Unknown STREAM_QUALITY '%s', using smooth\n", c->quality);
    }

    /* bits = pixels * fps * bits-per-pixel, so bitrate follows the resolution. */
    gint64 bitrate = (gint64) c->width * c->height * c->fps * bpp / 100;
    c->bitrate = env_int("VIDEO_BITRATE", (gint) MAX(bitrate, 500000));
    c->peak_bitrate = env_int("PEAK_BITRATE", (gint) ((gint64) c->bitrate * peak_pct / 100));
    c->gop = c->fps * env_int("GOP_SECONDS", env_int("KEYINT_SECONDS", 2));
}

static void load_config(Config *c)
{
    c->rtmp_uri = g_strdup_printf("rtmp://%s:%d/%s/%s live=1",
        env_str("STREAM_IP", "127.0.0.1"), env_int("STREAM_PORT", 1935),
        env_str("STREAM_APPLICATION", "live"), env_str("STREAM_KEY", "stream"));

    c->sensor_id = env_int("CAMERA_INDEX", 0);
    c->wb_mode = env_int("CAMERA_WBMODE", 1);
    c->tnr_mode = env_int("CAMERA_TNR_MODE", 1);
    c->tnr_strength = env_double("CAMERA_TNR_STRENGTH", 0.5);
    parse_resolution(c);
    parse_quality(c);

    c->audio = env_bool("AUDIO_ENABLED", FALSE);
    c->audio_device = g_strdup(env_str("AUDIO_DEVICE", "hw:2,0"));
    c->audio_bitrate = env_int("AUDIO_BITRATE", 96000);
    c->audio_rate = env_int("AUDIO_RATE", 48000);
    c->audio_channels = env_int("AUDIO_CHANNELS", 1);

    c->detection = env_bool("OBJECT_DETECTION", TRUE);
    const gchar *model = env_str("MODEL", "yolo26n_640x384.onnx");
    c->model_onnx = g_path_is_absolute(model) ? g_strdup(model) : g_build_filename("/models", model, NULL);
    gchar *stem = g_strndup(c->model_onnx, strlen(c->model_onnx) - (g_str_has_suffix(c->model_onnx, ".onnx") ? 5 : 0));
    c->model_engine = g_strdup_printf("%s_fp16.engine", stem);
    /* models/export.py writes <model>_labels.txt, so custom models carry their classes. */
    gchar *model_labels = g_strdup_printf("%s_labels.txt", stem);
    c->labels_path = g_strdup(env_str("MODEL_LABELS",
        g_file_test(model_labels, G_FILE_TEST_EXISTS) ? model_labels : "/models/labels.txt"));
    g_free(model_labels);
    g_free(stem);
    c->classes = g_strdup(env_str("CLASSES", ""));
    c->confidence = env_double("CONFIDENCE", 0.4);
    c->nms_iou = env_double("NMS_IOU", 0.45);
    c->interval = env_int("INFERENCE_INTERVAL", 0);
    c->tracker = g_ascii_strdown(env_str("TRACKER", "none"), -1);
    c->osd_mode = g_ascii_strcasecmp(env_str("OSD_MODE", "cpu"), "hw") == 0 ? 2 : 0;
    c->show_confidence = env_bool("SHOW_CONFIDENCE", TRUE);

    c->stats_interval = MAX(env_int("STATS_INTERVAL", 10), 1);
    c->watchdog_seconds = env_int("WATCHDOG_SECONDS", 20);
}

/* ------------------------------------------------------------------------- */
/* Model setup                                                                */
/* ------------------------------------------------------------------------- */

static gchar **labels;
static guint num_labels;

static gboolean load_labels(const gchar *path)
{
    gchar *contents = NULL;
    GError *error = NULL;
    if (!g_file_get_contents(path, &contents, NULL, &error)) {
        g_printerr("[ERROR] Cannot read labels: %s\n", error->message);
        g_error_free(error);
        return FALSE;
    }
    gchar **lines = g_strsplit(contents, "\n", -1);
    GPtrArray *array = g_ptr_array_new();
    for (gchar **line = lines; *line; line++) {
        g_strstrip(*line);
        if (**line)
            g_ptr_array_add(array, g_strdup(*line));
    }
    num_labels = MIN(array->len, MAX_CLASSES);
    g_ptr_array_add(array, NULL);
    labels = (gchar **) g_ptr_array_free(array, FALSE);
    g_strfreev(lines);
    g_free(contents);
    return num_labels > 0;
}

/* CLASSES accepts ids and/or names: "[0, 16]", "person,dog", "0, dog". */
static gboolean parse_classes(const gchar *spec, gboolean keep[MAX_CLASSES])
{
    /* Quotes are split on too: env files passed through Docker keep them. */
    gchar **tokens = g_strsplit_set(spec, "[],;\"'", -1);
    gboolean any = FALSE;
    for (gchar **token = tokens; *token; token++) {
        gchar *name = g_strstrip(*token);
        if (!*name)
            continue;

        gchar *end = NULL;
        gint64 id = g_ascii_strtoll(name, &end, 10);
        if (end && *end != '\0') {
            id = -1;
            for (guint i = 0; i < num_labels; i++)
                if (g_ascii_strcasecmp(labels[i], name) == 0)
                    id = i;
        }
        if (id < 0 || id >= num_labels) {
            g_printerr("[WARN] Ignoring unknown class '%s' in CLASSES\n", name);
            continue;
        }
        keep[id] = TRUE;
        any = TRUE;
    }
    g_strfreev(tokens);
    return any;
}

/* TensorRT engines are specific to the device + TensorRT version, so they are
 * built on the Jetson the first time a model is used and cached next to it. */
static gboolean ensure_engine(const Config *c)
{
    if (g_file_test(c->model_engine, G_FILE_TEST_EXISTS))
        return TRUE;
    if (!g_file_test(c->model_onnx, G_FILE_TEST_EXISTS)) {
        g_printerr("[ERROR] Model not found: %s\n", c->model_onnx);
        return FALSE;
    }

    g_print("[INFO] Building TensorRT FP16 engine for %s (one-time, ~10-15 minutes on a Nano)...\n", c->model_onnx);
    gchar *partial = g_strdup_printf("%s.partial", c->model_engine);
    gchar *onnx_arg = g_strdup_printf("--onnx=%s", c->model_onnx);
    gchar *save_arg = g_strdup_printf("--saveEngine=%s", partial);
    gchar *argv[] = { TRTEXEC, onnx_arg, save_arg, "--fp16", "--workspace=1024", NULL };
    gint status = -1;
    gchar *output = NULL;
    GError *error = NULL;
    gboolean ok = g_spawn_sync(NULL, argv, NULL, G_SPAWN_STDERR_TO_DEV_NULL, NULL, NULL, &output, NULL, &status, &error) &&
                  g_spawn_check_exit_status(status, &error) && g_rename(partial, c->model_engine) == 0;
    if (!ok) {
        /* trtexec logs everything to stdout; the reason is in the last lines. */
        gsize len = output ? strlen(output) : 0;
        g_printerr("%s\n[ERROR] Engine build failed: %s\n", len > 3000 ? output + len - 3000 : (output ? output : ""),
                   error ? error->message : "rename failed");
        g_clear_error(&error);
        g_unlink(partial);
    } else {
        g_print("[INFO] Engine saved to %s\n", c->model_engine);
    }
    g_free(output);
    g_free(partial);
    g_free(onnx_arg);
    g_free(save_arg);
    return ok;
}

static gchar *write_infer_config(const Config *c)
{
    gboolean keep[MAX_CLASSES] = { FALSE };
    GString *filter = g_string_new(NULL);
    if (parse_classes(c->classes, keep)) {
        for (guint i = 0; i < num_labels; i++)
            if (!keep[i])
                g_string_append_printf(filter, "%s%u", filter->len ? ";" : "", i);
    }

    gchar threshold[G_ASCII_DTOSTR_BUF_SIZE], iou[G_ASCII_DTOSTR_BUF_SIZE];
    g_ascii_dtostr(threshold, sizeof threshold, c->confidence);
    g_ascii_dtostr(iou, sizeof iou, c->nms_iou);

    gchar *config = g_strdup_printf(
        "[property]\n"
        "gpu-id=0\n"
        "net-scale-factor=0.0039215697906911373\n"
        "model-color-format=0\n"
        "onnx-file=%s\n"
        "model-engine-file=%s\n"
        "labelfile-path=%s\n"
        "batch-size=1\n"
        "network-mode=2\n"
        "num-detected-classes=%u\n"
        "interval=%d\n"
        "gie-unique-id=1\n"
        "process-mode=1\n"
        "network-type=0\n"
        /* NMS for anchor heads; NMS_IOU=0 disables it for NMS-free models. */
        "cluster-mode=%d\n"
        "maintain-aspect-ratio=1\n"
        "symmetric-padding=1\n"
        "output-blob-names=output\n"
        "parse-bbox-func-name=NvDsInferParseYolo\n"
        "custom-lib-path=" APP_DIR "/libnvdsinfer_custom_impl_yolo.so\n"
        "%s%s%s"
        "\n"
        "[class-attrs-all]\n"
        "pre-cluster-threshold=%s\n"
        "nms-iou-threshold=%s\n"
        "topk=100\n",
        c->model_onnx, c->model_engine, c->labels_path, num_labels, c->interval,
        c->nms_iou > 0 ? 2 : 4,
        filter->len ? "filter-out-class-ids=" : "", filter->str, filter->len ? "\n" : "",
        threshold, iou);

    gchar *path = g_build_filename(RUNTIME_DIR, "nvinfer.txt", NULL);
    g_mkdir_with_parents(RUNTIME_DIR, 0755);
    if (!g_file_set_contents(path, config, -1, NULL)) {
        g_printerr("[ERROR] Cannot write %s\n", path);
        g_clear_pointer(&path, g_free);
    }
    g_free(config);
    g_string_free(filter, TRUE);
    return path;
}

/* ------------------------------------------------------------------------- */
/* Pipeline                                                                   */
/* ------------------------------------------------------------------------- */

static gchar *build_pipeline_description(const Config *c, const gchar *infer_config)
{
    GString *d = g_string_new(NULL);

    g_string_append_printf(d,
        "nvarguscamerasrc name=camera sensor-id=%d bufapi-version=1 do-timestamp=true "
        "wbmode=%d tnr-mode=%d tnr-strength=%.2f ! "
        "video/x-raw(memory:NVMM),width=%d,height=%d,framerate=%d/1,format=NV12 ! ",
        c->sensor_id, c->wb_mode, c->tnr_mode, c->tnr_strength,
        c->capture_width, c->capture_height, c->fps);

    if (c->detection) {
        g_string_append_printf(d,
            "nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA,width=%d,height=%d ! "
            "mux_batch.sink_0 nvstreammux name=mux_batch batch-size=1 width=%d height=%d "
            "live-source=1 batched-push-timeout=%d ! "
            "nvinfer name=pgie config-file-path=%s ! ",
            c->width, c->height, c->width, c->height, 2000000 / c->fps, infer_config);

        if (g_str_equal(c->tracker, "iou") || g_str_equal(c->tracker, "nvdcf")) {
            g_string_append_printf(d,
                "nvtracker name=tracker ll-lib-file=" DS_LIB_DIR "/libnvds_nvmultiobjecttracker.so "
                "ll-config-file=" APP_DIR "/config/tracker_%s.yml tracker-width=640 tracker-height=384 "
                "display-tracking-id=0 ! ", c->tracker);
        }
        g_string_append_printf(d, "nvdsosd name=osd process-mode=%d ! ", c->osd_mode);
    }

    g_string_append_printf(d,
        "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12,width=%d,height=%d ! "
        "nvv4l2h264enc name=encoder bufapi-version=1 maxperf-enable=1 profile=4 "
        "control-rate=%d bitrate=%d peak-bitrate=%d preset-level=%d "
        "iframeinterval=%d idrinterval=%d insert-sps-pps=1 insert-vui=1 ! "
        "h264parse ! queue max-size-buffers=0 max-size-bytes=0 max-size-time=1000000000 ! flv.video ",
        c->width, c->height, c->control_rate, c->bitrate, c->peak_bitrate, c->preset_level, c->gop, c->gop);

    if (c->audio) {
        /* do-timestamp + provide-clock=false keep ALSA on the pipeline clock so
         * flvmux sees aligned audio/video timestamps. The first queue is large and
         * leaky so the mic never blocks while the camera is starting up. */
        g_string_append_printf(d,
            "alsasrc device=%s do-timestamp=true provide-clock=false ! "
            "audio/x-raw,format=S16LE,channels=%d,rate=%d ! "
            "queue max-size-buffers=0 max-size-bytes=0 max-size-time=2000000000 leaky=downstream ! "
            "audioconvert ! audioresample ! voaacenc bitrate=%d ! "
            "queue max-size-buffers=0 max-size-bytes=0 max-size-time=1000000000 ! flv.audio ",
            c->audio_device, c->audio_channels, c->audio_rate, c->audio_bitrate);
    }

    /* flvmux waits up to `latency` for the video pad: frames reach it ~150 ms
     * after capture (inference + encode) while audio is nearly instant, so a
     * short window interleaves audio ahead of video ("backwards dts").
     * Bounded, leaky network queue: when WiFi stalls, the oldest data is dropped
     * (recovering at the next keyframe) instead of stalling the camera. */
    g_string_append_printf(d,
        "flvmux name=flv streamable=true latency=500000000 ! "
        "queue name=netqueue max-size-buffers=0 max-size-bytes=0 max-size-time=%" G_GUINT64_FORMAT " leaky=downstream ! "
        "rtmpsink name=sink location=\"%s\" sync=false async=false",
        (guint64) (c->gop / c->fps) * GST_SECOND, c->rtmp_uri);

    return g_string_free(d, FALSE);
}

/* ------------------------------------------------------------------------- */
/* Probes, stats and watchdog                                                 */
/* ------------------------------------------------------------------------- */

static GstElement *pipeline;
static GMainLoop *loop;
static gint exit_code = 0;
static Config cfg;

static gint camera_frames, inferred_frames, encoded_frames, net_drops;
static gint64 upload_bytes;
static gint64 latency_sum_us, latency_count;
static gint64 last_output_us, started_us, shutdown_deadline_us;
static gint last_counts[MAX_CLASSES];
G_LOCK_DEFINE_STATIC(stats);

static const gdouble palette[][3] = {
    { 1.00, 0.22, 0.22 }, { 0.20, 0.80, 1.00 }, { 0.30, 1.00, 0.30 }, { 1.00, 0.80, 0.10 },
    { 1.00, 0.35, 1.00 }, { 0.10, 1.00, 0.80 }, { 1.00, 0.55, 0.10 }, { 0.60, 0.50, 1.00 },
};

/* Read and reset a counter (g_atomic_int_exchange needs GLib 2.74; 18.04 has 2.56). */
static gint take_count(gint *counter)
{
    gint value;
    do {
        value = g_atomic_int_get(counter);
    } while (!g_atomic_int_compare_and_exchange(counter, value, 0));
    return value;
}

static GstPadProbeReturn count_probe(GstPad *pad, GstPadProbeInfo *info, gpointer counter)
{
    g_atomic_int_inc((gint *) counter);
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn infer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(GST_PAD_PROBE_INFO_BUFFER(info));
    if (!batch)
        return GST_PAD_PROBE_OK;
    for (NvDsMetaList *l = batch->frame_meta_list; l; l = l->next)
        if (((NvDsFrameMeta *) l->data)->bInferDone)
            g_atomic_int_inc(&inferred_frames);
    return GST_PAD_PROBE_OK;
}

/* Colour boxes per class and draw "label 87%" tags. */
static GstPadProbeReturn osd_probe(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(GST_PAD_PROBE_INFO_BUFFER(info));
    if (!batch)
        return GST_PAD_PROBE_OK;

    gint counts[MAX_CLASSES] = { 0 };
    for (NvDsMetaList *lf = batch->frame_meta_list; lf; lf = lf->next) {
        NvDsFrameMeta *frame = (NvDsFrameMeta *) lf->data;
        for (NvDsMetaList *lo = frame->obj_meta_list; lo; lo = lo->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) lo->data;
            const gdouble *rgb = palette[obj->class_id % G_N_ELEMENTS(palette)];
            if (obj->class_id >= 0 && obj->class_id < MAX_CLASSES)
                counts[obj->class_id]++;

            NvOSD_RectParams *rect = &obj->rect_params;
            rect->border_width = 3;
            rect->border_color = (NvOSD_ColorParams) { rgb[0], rgb[1], rgb[2], 1.0 };
            rect->has_bg_color = 0;

            NvOSD_TextParams *text = &obj->text_params;
            g_free(text->display_text);
            if (cfg.show_confidence && obj->confidence > 0)
                text->display_text = g_strdup_printf(" %s %.0f%% ", obj->obj_label, obj->confidence * 100);
            else
                text->display_text = g_strdup_printf(" %s ", obj->obj_label);
            text->x_offset = (guint) MAX(rect->left, 0);
            text->y_offset = (guint) MAX(rect->top - 22, 0);
            text->font_params.font_name = "Sans Bold";
            text->font_params.font_size = 11;
            text->font_params.font_color = (NvOSD_ColorParams) { 0.0, 0.0, 0.0, 1.0 };
            text->set_bg_clr = 1;
            text->text_bg_clr = (NvOSD_ColorParams) { rgb[0], rgb[1], rgb[2], 0.9 };
        }
    }

    G_LOCK(stats);
    memcpy(last_counts, counts, sizeof counts);
    G_UNLOCK(stats);
    return GST_PAD_PROBE_OK;
}

/* Pipeline latency: the camera stamps buffers with the running time when it
 * pushes them, so (running time after encoding - PTS) is the time spent in
 * scaling, inference, tracking, drawing and encoding. */
static GstPadProbeReturn encoder_probe(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    g_atomic_int_inc(&encoded_frames);

    GstClock *clock = gst_element_get_clock(pipeline);
    if (clock && GST_BUFFER_PTS_IS_VALID(buffer)) {
        GstClockTime now = gst_clock_get_time(clock) - gst_element_get_base_time(pipeline);
        if (now > GST_BUFFER_PTS(buffer)) {
            G_LOCK(stats);
            latency_sum_us += (now - GST_BUFFER_PTS(buffer)) / GST_USECOND;
            latency_count++;
            G_UNLOCK(stats);
        }
    }
    if (clock)
        gst_object_unref(clock);
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn sink_probe(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    G_LOCK(stats);
    upload_bytes += gst_buffer_get_size(GST_PAD_PROBE_INFO_BUFFER(info));
    last_output_us = g_get_monotonic_time();
    G_UNLOCK(stats);
    return GST_PAD_PROBE_OK;
}

static void on_netqueue_overrun(GstElement *queue, gpointer data)
{
    g_atomic_int_inc(&net_drops);
}

static void add_probe(const gchar *element_name, const gchar *pad_name, GstPadProbeCallback callback, gpointer data)
{
    GstElement *element = gst_bin_get_by_name(GST_BIN(pipeline), element_name);
    if (!element)
        return;
    GstPad *pad = gst_element_get_static_pad(element, pad_name);
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, callback, data, NULL);
    gst_object_unref(pad);
    gst_object_unref(element);
}

static gboolean read_uint64s(const gchar *path, guint64 *values, gint n)
{
    gchar *contents = NULL;
    if (!g_file_get_contents(path, &contents, NULL, NULL))
        return FALSE;
    gchar *p = contents;
    if (!g_ascii_isdigit(*p))
        while (*p && !g_ascii_isspace(*p))
            p++; /* skip "cpu" label */
    for (gint i = 0; i < n; i++)
        values[i] = g_ascii_strtoull(p, &p, 10);
    g_free(contents);
    return TRUE;
}

static gboolean stats_tick(gpointer data)
{
    static guint ticks;
    static guint64 gpu_load_sum, cpu_prev[8], gpu_temp_max;
    guint64 value;

    if (read_uint64s("/sys/devices/57000000.gpu/load", &value, 1))
        gpu_load_sum += value;
    if (read_uint64s("/sys/devices/virtual/thermal/thermal_zone2/temp", &value, 1)) /* GPU-therm on the Nano */
        gpu_temp_max = MAX(gpu_temp_max, value);

    if (++ticks < (guint) cfg.stats_interval)
        return G_SOURCE_CONTINUE;

    guint64 cpu[8] = { 0 };
    gdouble cpu_pct = 0;
    if (read_uint64s("/proc/stat", cpu, 8)) {
        guint64 busy = 0, total = 0;
        for (gint i = 0; i < 8; i++) {
            guint64 delta = cpu[i] - cpu_prev[i];
            total += delta;
            if (i != 3 && i != 4) /* idle, iowait */
                busy += delta;
        }
        cpu_pct = total ? 100.0 * busy / total : 0;
        memcpy(cpu_prev, cpu, sizeof cpu);
    }

    gdouble secs = ticks;
    GString *objects = g_string_new(NULL);
    G_LOCK(stats);
    gint64 bytes = upload_bytes;
    gint64 latency_ms = latency_count ? latency_sum_us / latency_count / 1000 : -1;
    upload_bytes = latency_sum_us = latency_count = 0;
    for (guint i = 0; i < num_labels; i++)
        if (last_counts[i])
            g_string_append_printf(objects, "%s%d %s", objects->len ? ", " : "", last_counts[i], labels[i]);
    G_UNLOCK(stats);

    GString *line = g_string_new(NULL);
    g_string_append_printf(line, "[STATS] camera %.1f fps", take_count(&camera_frames) / secs);
    if (cfg.detection)
        g_string_append_printf(line, " | inference %.1f fps", take_count(&inferred_frames) / secs);
    g_string_append_printf(line, " | stream %.1f fps | latency %" G_GINT64_FORMAT " ms | upload %" G_GINT64_FORMAT " kbps",
                           take_count(&encoded_frames) / secs, latency_ms,
                           (gint64) (bytes * 8 / 1000 / secs));
    g_string_append_printf(line, " | net drops %d | CPU %.0f%% GPU %.0f%% %.0fC",
                           take_count(&net_drops), cpu_pct,
                           gpu_load_sum / 10.0 / secs, gpu_temp_max / 1000.0);
    if (cfg.detection)
        g_string_append_printf(line, " | %s", objects->len ? objects->str : "no objects");
    g_print("%s\n", line->str);

    g_string_free(line, TRUE);
    g_string_free(objects, TRUE);
    ticks = 0;
    gpu_load_sum = gpu_temp_max = 0;
    return G_SOURCE_CONTINUE;
}

/* Runs on its own thread so it still fires if a streaming thread is wedged in a
 * blocking socket write (librtmp cannot be interrupted) or teardown hangs. */
static gpointer watchdog_thread(gpointer data)
{
    for (;;) {
        g_usleep(G_USEC_PER_SEC);
        gint64 now = g_get_monotonic_time();

        G_LOCK(stats);
        gint64 last = last_output_us, deadline = shutdown_deadline_us;
        G_UNLOCK(stats);

        if (deadline && now > deadline) {
            g_printerr("[ERROR] Shutdown timed out, exiting\n");
            _exit(exit_code);
        }
        /* Allow time for engine load, camera start and the RTMP handshake. */
        gint64 reference = last ? last : started_us + 60 * G_USEC_PER_SEC;
        if (!deadline && cfg.watchdog_seconds > 0 && now - reference > cfg.watchdog_seconds * G_USEC_PER_SEC) {
            g_printerr("[ERROR] No data sent for %d s (network or camera stalled), exiting so Docker restarts the stream\n",
                       cfg.watchdog_seconds);
            _exit(3);
        }
    }
    return NULL;
}

static void quit(gint code)
{
    exit_code = code;
    G_LOCK(stats);
    if (!shutdown_deadline_us)
        shutdown_deadline_us = g_get_monotonic_time() + 10 * G_USEC_PER_SEC;
    G_UNLOCK(stats);
    g_main_loop_quit(loop);
}

static gboolean on_signal(gpointer data)
{
    g_print("[INFO] Shutdown signal received, stopping stream...\n");
    quit(0);
    return G_SOURCE_REMOVE;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GError *error = NULL;
    gchar *debug = NULL;

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR:
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("[ERROR] %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
            g_printerr("[ERROR] Debug info: %s\n", debug);
        quit(1);
        break;
    case GST_MESSAGE_WARNING:
        gst_message_parse_warning(msg, &error, &debug);
        g_printerr("[WARN] %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
        break;
    case GST_MESSAGE_EOS:
        g_printerr("[ERROR] Unexpected end of stream\n");
        quit(1);
        break;
    case GST_MESSAGE_STATE_CHANGED:
        if (msg->src == GST_OBJECT(pipeline)) {
            GstState old_state, new_state;
            gst_message_parse_state_changed(msg, &old_state, &new_state, NULL);
            if (new_state == GST_STATE_PLAYING)
                g_print("[INFO] Pipeline is PLAYING\n");
        }
        break;
    default:
        break;
    }
    g_clear_error(&error);
    g_free(debug);
    return TRUE;
}

/* ------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);
    load_config(&cfg);

    g_print("[INFO] Holly Stream (DeepStream) starting\n");
    g_print("  Capture:   %dx%d @ %d fps (sensor %d)\n", cfg.capture_width, cfg.capture_height, cfg.fps, cfg.sensor_id);
    g_print("  Output:    %dx%d, H.264 %s %d kbps (%s), GOP %d\n", cfg.width, cfg.height,
            cfg.control_rate ? "CBR" : "VBR", cfg.bitrate / 1000, cfg.quality, cfg.gop);
    g_print("  Audio:     %s\n", cfg.audio ? cfg.audio_device : "disabled");
    g_print("  Target:    %s\n", cfg.rtmp_uri);

    gchar *infer_config = NULL;
    if (cfg.detection) {
        g_print("  Model:     %s (confidence %.2f, NMS IoU %.2f, interval %d, tracker %s)\n", cfg.model_onnx,
                cfg.confidence, cfg.nms_iou, cfg.interval, cfg.tracker);
        if (!load_labels(cfg.labels_path) || !ensure_engine(&cfg) || !(infer_config = write_infer_config(&cfg)))
            return 2;
        g_print("  Classes:   %s (of %u in %s)\n", *cfg.classes ? cfg.classes : "all", num_labels, cfg.labels_path);
    } else {
        g_print("  Detection: disabled\n");
    }

    gchar *description = build_pipeline_description(&cfg, infer_config);
    g_print("[INFO] Pipeline: %s\n", description);

    GError *error = NULL;
    pipeline = gst_parse_launch(description, &error);
    if (!pipeline || error) {
        g_printerr("[ERROR] Could not build pipeline: %s\n", error ? error->message : "unknown error");
        return 2;
    }

    add_probe("camera", "src", count_probe, &camera_frames);
    add_probe("encoder", "src", encoder_probe, NULL);
    add_probe("sink", "sink", sink_probe, NULL);
    if (cfg.detection) {
        add_probe("pgie", "src", infer_probe, NULL);
        add_probe("osd", "sink", osd_probe, NULL);
    }
    GstElement *netqueue = gst_bin_get_by_name(GST_BIN(pipeline), "netqueue");
    g_signal_connect(netqueue, "overrun", G_CALLBACK(on_netqueue_overrun), NULL);
    gst_object_unref(netqueue);

    loop = g_main_loop_new(NULL, FALSE);
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, NULL);
    gst_object_unref(bus);
    g_unix_signal_add(SIGINT, on_signal, NULL);
    g_unix_signal_add(SIGTERM, on_signal, NULL);
    g_timeout_add_seconds(1, stats_tick, NULL);

    started_us = g_get_monotonic_time();
    g_thread_new("watchdog", watchdog_thread, NULL);

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        g_printerr("[ERROR] Pipeline failed to start (is the RTMP server reachable?)\n");
        exit_code = 1;
    } else {
        g_main_loop_run(loop);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    g_print("[INFO] Stopped (exit code %d)\n", exit_code);
    return exit_code;
}
