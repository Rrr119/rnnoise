/* Copyright (c) 2026 RNNoise contributors */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "rnnoise.h"

#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#define RNNOISE_RATE 48000
#define RNNOISE_FRAME 480
#define RESAMPLER_TAPS 16
#define RESAMPLER_HALF (RESAMPLER_TAPS/2)
#define RESAMPLER_CAP 32768
#define MAX_CHANNELS 32
#define MAX_RATE 768000
#define DEFAULT_QUEUE_FRAMES 32
#define PI 3.14159265358979323846

typedef enum {
  ENC_U8,
  ENC_S16LE,
  ENC_S16BE,
  ENC_S24LE,
  ENC_S24BE,
  ENC_S32LE,
  ENC_S32BE,
  ENC_F32LE,
  ENC_F32BE
} SampleEncoding;

typedef struct {
  uint32_t sample_rate;
  int channels;
  SampleEncoding encoding;
} AudioFormat;

typedef struct {
  FILE *file;
  unsigned char prefix[12];
  size_t prefix_len;
  size_t prefix_pos;
} ByteStream;

typedef struct {
  AudioFormat format;
  uint64_t block_align;
  uint64_t data_bytes;
  int have_format;
} WavInfo;

typedef struct {
  ByteStream stream;
  AudioFormat format;
  uint64_t block_align;
  uint64_t data_bytes;
  uint64_t total_frames;
  int is_wav;
  int owns_file;
  const char *path;
} AudioInput;

typedef struct {
  AudioInput *input;
  unsigned char buffer[65536];
} PcmReader;

typedef struct {
  uint32_t src_rate;
  uint32_t dst_rate;
  double ratio;
  float samples[RESAMPLER_CAP];
  size_t head;
  size_t count;
  int64_t first_index;
  uint64_t total_in;
  uint64_t total_out;
  uint64_t target_out;
  int eof;
} Resampler;

typedef struct {
  float samples[RNNOISE_FRAME];
  uint64_t input_samples_total;
  float vad;
  int has_vad;
  int is_end;
} Frame;

typedef struct {
  Frame *frames;
  int capacity;
  int head;
  int count;
  int closed;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} FrameQueue;

typedef struct {
  pthread_mutex_t mutex;
  int failed;
} SharedState;

typedef struct {
  const char *input_path;
  const char *output_path;
  const char *vad_path;
  const char *model_path;
  const char *input_format;
  const char *output_format;
  const char *sample_format;
  const char *output_sample_format;
  uint32_t raw_rate;
  int raw_channels;
  int queue_frames;
  int quiet;
} Config;

typedef struct {
  AudioInput *input;
  FrameQueue *output_queue;
  SharedState *shared;
  uint64_t source_frames;
} ReaderContext;

typedef struct {
  FrameQueue *input_queue;
  FrameQueue *output_queue;
  SharedState *shared;
  RNNModel *model;
} WorkerContext;

typedef struct {
  FrameQueue *input_queue;
  SharedState *shared;
  FILE *audio_file;
  FILE *vad_file;
  AudioFormat format;
  int output_wav;
  Resampler resampler;
  int bypass_resampler;
  float *chunk;
  size_t chunk_capacity;
  float *hold;
  size_t hold_capacity;
  size_t hold_length;
  unsigned char *encode_buffer;
  size_t encode_capacity;
  uint64_t clean_samples;
  uint64_t written_frames;
  uint64_t vad_rows;
} WriterContext;

static int read_u16le(const unsigned char *p) {
  return p[0] | (p[1] << 8);
}

static uint32_t read_u32le(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int sample_bytes(SampleEncoding encoding) {
  switch (encoding) {
    case ENC_U8: return 1;
    case ENC_S16LE:
    case ENC_S16BE: return 2;
    case ENC_S24LE:
    case ENC_S24BE: return 3;
    case ENC_S32LE:
    case ENC_S32BE:
    case ENC_F32LE:
    case ENC_F32BE: return 4;
  }
  return 0;
}

static int sample_encoding_is_float(SampleEncoding encoding) {
  return encoding == ENC_F32LE || encoding == ENC_F32BE;
}

static SampleEncoding little_endian_encoding(SampleEncoding encoding) {
  switch (encoding) {
    case ENC_S16BE: return ENC_S16LE;
    case ENC_S24BE: return ENC_S24LE;
    case ENC_S32BE: return ENC_S32LE;
    case ENC_F32BE: return ENC_F32LE;
    default: return encoding;
  }
}

static float decode_sample(const unsigned char *p, SampleEncoding encoding) {
  int32_t value;
  uint32_t bits;
  float sample;
  switch (encoding) {
    case ENC_U8:
      return (float)((int)p[0] - 128) * 256.0f;
    case ENC_S16LE:
      value = (int16_t)(p[0] | (p[1] << 8));
      return (float)value;
    case ENC_S16BE:
      value = (int16_t)((p[0] << 8) | p[1]);
      return (float)value;
    case ENC_S24LE:
      value = p[0] | (p[1] << 8) | (p[2] << 16);
      if (value & 0x00800000) value |= (int32_t)0xff000000;
      return (float)value / 256.0f;
    case ENC_S24BE:
      value = (p[0] << 16) | (p[1] << 8) | p[2];
      if (value & 0x00800000) value |= (int32_t)0xff000000;
      return (float)value / 256.0f;
    case ENC_S32LE:
      bits = read_u32le(p);
      memcpy(&value, &bits, sizeof(value));
      return (float)value / 65536.0f;
    case ENC_S32BE:
      bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
             ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
      bits = ((bits & 0x000000ffU) << 24) |
             ((bits & 0x0000ff00U) << 8) |
             ((bits & 0x00ff0000U) >> 8) |
             ((bits & 0xff000000U) >> 24);
      memcpy(&value, &bits, sizeof(value));
      return (float)value / 65536.0f;
    case ENC_F32LE:
      bits = read_u32le(p);
      memcpy(&sample, &bits, sizeof(sample));
      return sample * 32768.0f;
    case ENC_F32BE:
      bits = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
             ((uint32_t)p[2] << 8) | (uint32_t)p[3];
      memcpy(&sample, &bits, sizeof(sample));
      return sample * 32768.0f;
  }
  return 0;
}

static double clamp_double(double value, double low, double high) {
  if (!(value == value)) return 0;
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

static void encode_sample(float sample, SampleEncoding encoding,
                          unsigned char *p) {
  double value = sample;
  int32_t integer;
  uint32_t bits;
  float floating;
  switch (encoding) {
    case ENC_U8:
      integer = (int32_t)lrint(clamp_double(value / 256.0 + 128.0, 0, 255));
      p[0] = (unsigned char)integer;
      break;
    case ENC_S16LE:
      integer = (int32_t)lrint(clamp_double(value, -32768, 32767));
      p[0] = (unsigned char)(integer & 0xff);
      p[1] = (unsigned char)((integer >> 8) & 0xff);
      break;
    case ENC_S16BE:
      integer = (int32_t)lrint(clamp_double(value, -32768, 32767));
      p[0] = (unsigned char)((integer >> 8) & 0xff);
      p[1] = (unsigned char)(integer & 0xff);
      break;
    case ENC_S24LE:
      integer = (int32_t)lrint(clamp_double(value * 256.0,
                                           -8388608, 8388607));
      p[0] = (unsigned char)(integer & 0xff);
      p[1] = (unsigned char)((integer >> 8) & 0xff);
      p[2] = (unsigned char)((integer >> 16) & 0xff);
      break;
    case ENC_S24BE:
      integer = (int32_t)lrint(clamp_double(value * 256.0,
                                           -8388608, 8388607));
      p[0] = (unsigned char)((integer >> 16) & 0xff);
      p[1] = (unsigned char)((integer >> 8) & 0xff);
      p[2] = (unsigned char)(integer & 0xff);
      break;
    case ENC_S32LE:
      integer = (int32_t)lrint(clamp_double(value * 65536.0,
                                           -2147483648.0, 2147483647.0));
      bits = (uint32_t)integer;
      p[0] = (unsigned char)(bits & 0xff);
      p[1] = (unsigned char)((bits >> 8) & 0xff);
      p[2] = (unsigned char)((bits >> 16) & 0xff);
      p[3] = (unsigned char)((bits >> 24) & 0xff);
      break;
    case ENC_S32BE:
      integer = (int32_t)lrint(clamp_double(value * 65536.0,
                                           -2147483648.0, 2147483647.0));
      bits = (uint32_t)integer;
      p[0] = (unsigned char)((bits >> 24) & 0xff);
      p[1] = (unsigned char)((bits >> 16) & 0xff);
      p[2] = (unsigned char)((bits >> 8) & 0xff);
      p[3] = (unsigned char)(bits & 0xff);
      break;
    case ENC_F32LE:
      floating = (float)(clamp_double(value, -32768, 32767) / 32768.0);
      memcpy(&bits, &floating, sizeof(bits));
      p[0] = (unsigned char)(bits & 0xff);
      p[1] = (unsigned char)((bits >> 8) & 0xff);
      p[2] = (unsigned char)((bits >> 16) & 0xff);
      p[3] = (unsigned char)((bits >> 24) & 0xff);
      break;
    case ENC_F32BE:
      floating = (float)(clamp_double(value, -32768, 32767) / 32768.0);
      memcpy(&bits, &floating, sizeof(bits));
      p[0] = (unsigned char)((bits >> 24) & 0xff);
      p[1] = (unsigned char)((bits >> 16) & 0xff);
      p[2] = (unsigned char)((bits >> 8) & 0xff);
      p[3] = (unsigned char)(bits & 0xff);
      break;
  }
}

static int parse_sample_format(const char *name, SampleEncoding *encoding) {
  if (strcmp(name, "u8") == 0) *encoding = ENC_U8;
  else if (strcmp(name, "s16") == 0 || strcmp(name, "s16le") == 0)
    *encoding = ENC_S16LE;
  else if (strcmp(name, "s16be") == 0) *encoding = ENC_S16BE;
  else if (strcmp(name, "s24") == 0 || strcmp(name, "s24le") == 0)
    *encoding = ENC_S24LE;
  else if (strcmp(name, "s24be") == 0) *encoding = ENC_S24BE;
  else if (strcmp(name, "s32") == 0 || strcmp(name, "s32le") == 0)
    *encoding = ENC_S32LE;
  else if (strcmp(name, "s32be") == 0) *encoding = ENC_S32BE;
  else if (strcmp(name, "f32") == 0 || strcmp(name, "f32le") == 0)
    *encoding = ENC_F32LE;
  else if (strcmp(name, "f32be") == 0) *encoding = ENC_F32BE;
  else return -1;
  return 0;
}

static size_t stream_read(ByteStream *stream, void *destination, size_t length) {
  size_t done = 0;
  unsigned char *dst = (unsigned char *)destination;
  if (stream->prefix_pos < stream->prefix_len) {
    size_t available = stream->prefix_len - stream->prefix_pos;
    size_t take = available < length ? available : length;
    memcpy(dst, stream->prefix + stream->prefix_pos, take);
    stream->prefix_pos += take;
    done += take;
  }
  while (done < length) {
    size_t count = fread(dst + done, 1, length - done, stream->file);
    if (count == 0) break;
    done += count;
  }
  return done;
}

static int stream_read_exact(ByteStream *stream, void *destination,
                             size_t length) {
  return stream_read(stream, destination, length) == length;
}

static int stream_skip(ByteStream *stream, uint64_t length) {
  unsigned char buffer[4096];
  while (length > 0) {
    size_t take = length > sizeof(buffer) ? sizeof(buffer) : (size_t)length;
    if (!stream_read_exact(stream, buffer, take)) return -1;
    length -= take;
  }
  return 0;
}

static int parse_wav_format(const unsigned char *data, size_t length,
                            WavInfo *info) {
  uint64_t block_align;
  int format;
  int bits;
  int channels;
  SampleEncoding encoding;
  if (length < 16) return -1;
  format = read_u16le(data);
  channels = read_u16le(data + 2);
  info->format.sample_rate = read_u32le(data + 4);
  block_align = read_u16le(data + 12);
  bits = read_u16le(data + 14);
  if (format == 0xfffe) {
    if (length < 40) return -1;
    format = (int)read_u32le(data + 24);
  }
  if (channels <= 0 || channels > MAX_CHANNELS) return -1;
  if (info->format.sample_rate == 0 ||
      info->format.sample_rate > MAX_RATE) return -1;
  if (format == 1) {
    switch (bits) {
      case 8: encoding = ENC_U8; break;
      case 16: encoding = ENC_S16LE; break;
      case 24: encoding = ENC_S24LE; break;
      case 32: encoding = ENC_S32LE; break;
      default: return -1;
    }
  } else if (format == 3 && bits == 32) {
    encoding = ENC_F32LE;
  } else {
    return -1;
  }
  if (block_align == 0) {
    block_align = (uint64_t)channels * (uint64_t)sample_bytes(encoding);
  }
  if (block_align < (uint64_t)channels * (uint64_t)sample_bytes(encoding))
    return -1;
  info->format.channels = channels;
  info->format.encoding = encoding;
  info->block_align = block_align;
  info->have_format = 1;
  return 0;
}

static int parse_wav(ByteStream *stream, WavInfo *info) {
  unsigned char header[12];
  unsigned char chunk_header[8];
  unsigned char format_data[40];
  uint32_t chunk_size;
  size_t take;
  memset(info, 0, sizeof(*info));
  if (!stream_read_exact(stream, header, sizeof(header))) return -1;
  if (memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0)
    return -1;
  while (1) {
    if (!stream_read_exact(stream, chunk_header, sizeof(chunk_header)))
      return -1;
    chunk_size = read_u32le(chunk_header + 4);
    if (memcmp(chunk_header, "fmt ", 4) == 0) {
      if (!info->have_format && chunk_size >= 16) {
        take = chunk_size < sizeof(format_data) ? chunk_size :
               sizeof(format_data);
        if (!stream_read_exact(stream, format_data, take)) return -1;
        if (parse_wav_format(format_data, take, info) != 0) return -1;
        if (take < chunk_size &&
            stream_skip(stream, chunk_size - take) != 0) return -1;
      } else {
        if (stream_skip(stream, chunk_size) != 0) return -1;
      }
    } else if (memcmp(chunk_header, "data", 4) == 0) {
      if (!info->have_format) return -1;
      info->data_bytes = chunk_size == UINT32_MAX ? UINT64_MAX : chunk_size;
      return 0;
    } else {
      if (stream_skip(stream, chunk_size) != 0) return -1;
    }
    if (chunk_size & 1) {
      unsigned char padding;
      if (!stream_read_exact(stream, &padding, 1)) return -1;
    }
  }
}

static int open_audio_input(const Config *config, AudioInput *input,
                            char *error, size_t error_size) {
  struct stat status;
  unsigned char magic[12];
  size_t magic_length = 0;
  int auto_format = strcmp(config->input_format, "auto") == 0;
  int want_wav = strcmp(config->input_format, "wav") == 0;
  int is_regular = 0;
  memset(input, 0, sizeof(*input));
  input->path = config->input_path;
  if (strcmp(config->input_path, "-") == 0) {
    input->stream.file = stdin;
  } else {
    input->stream.file = fopen(config->input_path, "rb");
    input->owns_file = 1;
  }
  if (input->stream.file == NULL) {
    snprintf(error, error_size, "cannot open '%s': %s", config->input_path,
             strerror(errno));
    return -1;
  }
  if (fstat(fileno(input->stream.file), &status) == 0) {
    is_regular = S_ISREG(status.st_mode);
  }
  if (auto_format) {
    magic_length = fread(magic, 1, sizeof(magic), input->stream.file);
    memcpy(input->stream.prefix, magic, magic_length);
    input->stream.prefix_len = magic_length;
    want_wav = magic_length >= 12 && memcmp(magic, "RIFF", 4) == 0 &&
               memcmp(magic + 8, "WAVE", 4) == 0;
  }
  if (want_wav) {
    WavInfo info;
    if (parse_wav(&input->stream, &info) != 0) {
      snprintf(error, error_size, "'%s' is not a supported WAV file",
               config->input_path);
      return -1;
    }
    input->is_wav = 1;
    input->format = info.format;
    input->block_align = info.block_align;
    input->data_bytes = info.data_bytes;
    if (info.data_bytes == UINT64_MAX) {
      input->total_frames = UINT64_MAX;
    } else {
      input->total_frames = info.data_bytes / info.block_align;
    }
  } else {
    SampleEncoding encoding;
    uint64_t total_bytes;
    if (strcmp(config->input_format, "raw") != 0 && !auto_format) {
      snprintf(error, error_size, "unknown input format '%s'",
               config->input_format);
      return -1;
    }
    if (parse_sample_format(config->sample_format, &encoding) != 0) {
      snprintf(error, error_size, "unknown sample format '%s'",
               config->sample_format);
      return -1;
    }
    input->is_wav = 0;
    input->format.sample_rate = config->raw_rate;
    input->format.channels = config->raw_channels;
    input->format.encoding = encoding;
    input->block_align = (uint64_t)config->raw_channels *
                         (uint64_t)sample_bytes(encoding);
    if (input->block_align == 0) {
      snprintf(error, error_size, "invalid raw block alignment");
      return -1;
    }
    total_bytes = is_regular ? (uint64_t)status.st_size : UINT64_MAX;
    input->data_bytes = total_bytes;
    input->total_frames = total_bytes == UINT64_MAX ? UINT64_MAX :
                          total_bytes / input->block_align;
  }
  if (input->format.channels <= 0 ||
      input->format.channels > MAX_CHANNELS ||
      input->format.sample_rate == 0 ||
      input->format.sample_rate > MAX_RATE) {
    snprintf(error, error_size, "unsupported input audio format");
    return -1;
  }
  return 0;
}

static void close_audio_input(AudioInput *input) {
  if (input->owns_file && input->stream.file != NULL)
    fclose(input->stream.file);
}

static void pcm_reader_init(PcmReader *reader, AudioInput *input) {
  reader->input = input;
}

static int pcm_read(PcmReader *reader, float *mono, size_t max_frames,
                    size_t *frames_read, int *eof) {
  AudioInput *input = reader->input;
  size_t bytes_per_sample = (size_t)sample_bytes(input->format.encoding);
  size_t block_align = (size_t)input->block_align;
  size_t wanted_bytes;
  size_t wanted_frames;
  size_t got_bytes;
  size_t got_frames;
  size_t frame;
  size_t channel;
  *frames_read = 0;
  *eof = 0;
  if (max_frames == 0) return 0;
  wanted_frames = max_frames;
  if (wanted_frames * block_align > sizeof(reader->buffer))
    wanted_frames = sizeof(reader->buffer) / block_align;
  if (wanted_frames == 0) return -1;
  wanted_bytes = wanted_frames * block_align;
  if (input->data_bytes != UINT64_MAX && input->data_bytes < wanted_bytes)
    wanted_bytes = (size_t)input->data_bytes;
  if (wanted_bytes == 0) {
    *eof = 1;
    return 0;
  }
  got_bytes = stream_read(&input->stream, reader->buffer, wanted_bytes);
  if (got_bytes == 0) {
    *eof = 1;
    return 0;
  }
  if (got_bytes % block_align != 0) return -1;
  got_frames = got_bytes / block_align;
  for (frame = 0; frame < got_frames; frame++) {
    const unsigned char *source =
        reader->buffer + frame * block_align;
    double sum = 0;
    for (channel = 0; channel < (size_t)input->format.channels; channel++) {
      sum += decode_sample(source + channel * bytes_per_sample,
                           input->format.encoding);
    }
    mono[frame] = (float)(sum / input->format.channels);
  }
  if (input->data_bytes != UINT64_MAX) {
    input->data_bytes -= got_bytes;
    if (input->data_bytes == 0) *eof = 1;
  } else if (got_bytes < wanted_bytes) {
    *eof = 1;
  }
  *frames_read = got_frames;
  return 0;
}

static void resampler_init(Resampler *resampler, uint32_t src_rate,
                           uint32_t dst_rate) {
  memset(resampler, 0, sizeof(*resampler));
  resampler->src_rate = src_rate;
  resampler->dst_rate = dst_rate;
  resampler->ratio = (double)src_rate / (double)dst_rate;
  resampler->target_out = UINT64_MAX;
}

static void resampler_discard(Resampler *resampler) {
  int64_t next_position;
  int64_t keep_from;
  size_t discard;
  next_position = (int64_t)floor((double)resampler->total_out *
                                 resampler->ratio);
  keep_from = next_position - RESAMPLER_HALF + 1;
  if (keep_from < 0) keep_from = 0;
  if (resampler->first_index >= keep_from) return;
  discard = (size_t)(keep_from - resampler->first_index);
  if (discard > resampler->count) discard = resampler->count;
  resampler->head += discard;
  resampler->count -= discard;
  resampler->first_index += (int64_t)discard;
  if (resampler->count == 0) {
    resampler->head = 0;
    resampler->first_index = (int64_t)resampler->total_in;
  } else if (resampler->head > RESAMPLER_CAP / 2) {
    memmove(resampler->samples, resampler->samples + resampler->head,
            resampler->count * sizeof(float));
    resampler->head = 0;
  }
}

static int resampler_feed(Resampler *resampler, const float *samples,
                          size_t count) {
  resampler_discard(resampler);
  if (resampler->count + count > RESAMPLER_CAP) return -1;
  if (resampler->head + resampler->count + count > RESAMPLER_CAP) {
    memmove(resampler->samples, resampler->samples + resampler->head,
            resampler->count * sizeof(float));
    resampler->head = 0;
  }
  if (resampler->count == 0)
    resampler->first_index = (int64_t)resampler->total_in;
  memcpy(resampler->samples + resampler->head + resampler->count, samples,
         count * sizeof(float));
  resampler->count += count;
  resampler->total_in += count;
  return 0;
}

static float resampler_source_sample(const Resampler *resampler,
                                     int64_t index) {
  size_t offset;
  if (resampler->total_in == 0) return 0;
  if (index < 0) index = 0;
  if ((uint64_t)index >= resampler->total_in)
    index = (int64_t)resampler->total_in - 1;
  if (index < resampler->first_index)
    index = resampler->first_index;
  offset = (size_t)(index - resampler->first_index);
  if (offset >= resampler->count)
    offset = resampler->count == 0 ? 0 : resampler->count - 1;
  return resampler->samples[resampler->head + offset];
}

static int resampler_can_read(const Resampler *resampler) {
  int64_t center;
  if (resampler->total_in == 0) return 0;
  if (resampler->eof) return resampler->total_out < resampler->target_out;
  center = (int64_t)floor((double)resampler->total_out * resampler->ratio);
  return center + RESAMPLER_HALF < (int64_t)resampler->total_in;
}

static double sinc(double value) {
  if (fabs(value) < 1e-12) return 1;
  return sin(PI * value) / (PI * value);
}

static float resampler_read(Resampler *resampler) {
  double position = (double)resampler->total_out * resampler->ratio;
  int64_t center = (int64_t)floor(position);
  double fraction = position - (double)center;
  double cutoff = 1;
  double weighted = 0;
  double weight_sum = 0;
  int tap;
  if (resampler->dst_rate < resampler->src_rate)
    cutoff = 0.95 * (double)resampler->dst_rate / resampler->src_rate;
  for (tap = -RESAMPLER_HALF + 1; tap <= RESAMPLER_HALF; tap++) {
    int64_t index = center + tap;
    double distance = (double)index - position;
    double window = 0;
    double weight;
    if (fabs(distance) < RESAMPLER_HALF)
      window = 0.5 + 0.5 * cos(PI * distance / RESAMPLER_HALF);
    weight = cutoff * sinc(cutoff * distance) * window;
    weighted += (double)resampler_source_sample(resampler, index) * weight;
    weight_sum += weight;
  }
  (void)fraction;
  resampler->total_out++;
  resampler_discard(resampler);
  if (fabs(weight_sum) < 1e-12)
    return resampler_source_sample(resampler, center);
  return (float)(weighted / weight_sum);
}

static void resampler_finish(Resampler *resampler, uint64_t total_in) {
  resampler->eof = 1;
  resampler->target_out = (uint64_t)llround((double)total_in *
                                            (double)resampler->dst_rate /
                                            (double)resampler->src_rate);
  if (resampler->target_out < resampler->total_out)
    resampler->target_out = resampler->total_out;
}

static int queue_init(FrameQueue *queue, int capacity) {
  memset(queue, 0, sizeof(*queue));
  queue->frames = (Frame *)calloc((size_t)capacity, sizeof(Frame));
  if (queue->frames == NULL) return -1;
  queue->capacity = capacity;
  if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
    free(queue->frames);
    queue->frames = NULL;
    return -1;
  }
  if (pthread_cond_init(&queue->cond, NULL) != 0) {
    pthread_mutex_destroy(&queue->mutex);
    free(queue->frames);
    queue->frames = NULL;
    return -1;
  }
  return 0;
}

static void queue_destroy(FrameQueue *queue) {
  if (queue->frames != NULL) {
    pthread_cond_destroy(&queue->cond);
    pthread_mutex_destroy(&queue->mutex);
    free(queue->frames);
    queue->frames = NULL;
  }
}

static void queue_close(FrameQueue *queue) {
  pthread_mutex_lock(&queue->mutex);
  queue->closed = 1;
  pthread_cond_broadcast(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);
}

static int queue_push(FrameQueue *queue, const Frame *frame) {
  int tail;
  pthread_mutex_lock(&queue->mutex);
  while (queue->count == queue->capacity && !queue->closed)
    pthread_cond_wait(&queue->cond, &queue->mutex);
  if (queue->closed) {
    pthread_mutex_unlock(&queue->mutex);
    return -1;
  }
  tail = (queue->head + queue->count) % queue->capacity;
  queue->frames[tail] = *frame;
  queue->count++;
  pthread_cond_broadcast(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);
  return 0;
}

static int queue_pop(FrameQueue *queue, Frame *frame) {
  pthread_mutex_lock(&queue->mutex);
  while (queue->count == 0 && !queue->closed)
    pthread_cond_wait(&queue->cond, &queue->mutex);
  if (queue->count == 0) {
    pthread_mutex_unlock(&queue->mutex);
    return -1;
  }
  *frame = queue->frames[queue->head];
  queue->head = (queue->head + 1) % queue->capacity;
  queue->count--;
  pthread_cond_broadcast(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);
  return 0;
}

static int shared_fail(SharedState *shared, const char *format, ...) {
  va_list arguments;
  pthread_mutex_lock(&shared->mutex);
  if (!shared->failed) {
    shared->failed = 1;
    va_start(arguments, format);
    fprintf(stderr, "rnnoise_stream: ");
    vfprintf(stderr, format, arguments);
    fprintf(stderr, "\n");
    va_end(arguments);
  }
  pthread_mutex_unlock(&shared->mutex);
  return -1;
}

static int shared_has_failed(SharedState *shared) {
  int failed;
  pthread_mutex_lock(&shared->mutex);
  failed = shared->failed;
  pthread_mutex_unlock(&shared->mutex);
  return failed;
}

typedef struct {
  float samples[RNNOISE_FRAME];
  size_t length;
  uint64_t frames;
} FrameBuilder;

static int push_builder_frame(ReaderContext *context, FrameBuilder *builder) {
  Frame frame;
  if (builder->length == 0) return 0;
  memset(&frame, 0, sizeof(frame));
  memcpy(frame.samples, builder->samples,
         builder->length * sizeof(float));
  if (queue_push(context->output_queue, &frame) != 0) return -1;
  builder->length = 0;
  return 0;
}

static int add_reader_samples(ReaderContext *context, FrameBuilder *builder,
                              const float *samples, size_t count) {
  size_t offset = 0;
  while (offset < count) {
    size_t available = RNNOISE_FRAME - builder->length;
    size_t take = count - offset < available ? count - offset : available;
    memcpy(builder->samples + builder->length, samples + offset,
           take * sizeof(float));
    builder->length += take;
    offset += take;
    if (builder->length == RNNOISE_FRAME) {
      if (push_builder_frame(context, builder) != 0) return -1;
    }
  }
  return 0;
}

static void *reader_thread(void *argument) {
  ReaderContext *context = (ReaderContext *)argument;
  PcmReader reader;
  FrameBuilder builder;
  Resampler resampler;
  Frame end_frame;
  float chunk[2048];
  size_t chunk_frames;
  size_t frames_read;
  int eof = 0;
  int bypass = context->input->format.sample_rate == RNNOISE_RATE;
  memset(&builder, 0, sizeof(builder));
  pcm_reader_init(&reader, context->input);
  if (!bypass)
    resampler_init(&resampler, context->input->format.sample_rate,
                   RNNOISE_RATE);
  chunk_frames = context->input->format.sample_rate / 100;
  if (chunk_frames == 0) chunk_frames = 1;
  if (chunk_frames > sizeof(chunk) / sizeof(chunk[0]))
    chunk_frames = sizeof(chunk) / sizeof(chunk[0]);
  while (!eof) {
    size_t index;
    if (pcm_read(&reader, chunk, chunk_frames, &frames_read, &eof) != 0) {
      shared_fail(context->shared, "failed to read input audio");
      goto fail;
    }
    context->source_frames += frames_read;
    if (bypass) {
      if (add_reader_samples(context, &builder, chunk, frames_read) != 0) {
        shared_fail(context->shared, "inference queue closed");
        goto fail;
      }
    } else {
      if (frames_read > 0 &&
          resampler_feed(&resampler, chunk, frames_read) != 0) {
        shared_fail(context->shared, "internal resampler buffer overflow");
        goto fail;
      }
      while (resampler_can_read(&resampler)) {
        for (index = 0; index < RNNOISE_FRAME; index++)
          chunk[index] = resampler_read(&resampler);
        if (add_reader_samples(context, &builder, chunk, RNNOISE_FRAME) != 0) {
          shared_fail(context->shared, "inference queue closed");
          goto fail;
        }
      }
    }
  }
  if (!bypass) {
    uint64_t target;
    resampler_finish(&resampler, context->source_frames);
    target = resampler.target_out;
    while (resampler.total_out < target) {
      size_t index;
      for (index = 0; index < RNNOISE_FRAME &&
                      resampler.total_out < target; index++)
        chunk[index] = resampler_read(&resampler);
      if (add_reader_samples(context, &builder, chunk, index) != 0) {
        shared_fail(context->shared, "inference queue closed");
        goto fail;
      }
    }
  }
  if (builder.length > 0) {
    if (push_builder_frame(context, &builder) != 0) {
      shared_fail(context->shared, "inference queue closed");
      goto fail;
    }
  }
  memset(&end_frame, 0, sizeof(end_frame));
  end_frame.is_end = 1;
  end_frame.input_samples_total = context->source_frames;
  if (queue_push(context->output_queue, &end_frame) != 0) {
    shared_fail(context->shared, "inference queue closed");
    goto fail;
  }
  queue_close(context->output_queue);
  return NULL;

fail:
  queue_close(context->output_queue);
  return NULL;
}

static int push_worker_result(WorkerContext *context, const float *samples,
                              float vad, int has_vad, int is_end,
                              uint64_t input_samples_total) {
  Frame frame;
  memset(&frame, 0, sizeof(frame));
  if (!is_end) memcpy(frame.samples, samples, sizeof(frame.samples));
  frame.vad = vad;
  frame.has_vad = has_vad;
  frame.is_end = is_end;
  frame.input_samples_total = input_samples_total;
  return queue_push(context->output_queue, &frame);
}

static void *worker_thread(void *argument) {
  WorkerContext *context = (WorkerContext *)argument;
  DenoiseState *state;
  Frame input_frame;
  float last_samples[RNNOISE_FRAME];
  float output[RNNOISE_FRAME];
  float previous_vad = 0;
  int have_previous = 0;
  memset(&input_frame, 0, sizeof(input_frame));
  state = rnnoise_create(context->model);
  if (state == NULL) {
    shared_fail(context->shared, "failed to initialize RNNoise");
    queue_close(context->input_queue);
    queue_close(context->output_queue);
    return NULL;
  }
  while (queue_pop(context->input_queue, &input_frame) == 0) {
    float vad;
    if (input_frame.is_end) {
      if (have_previous) {
        vad = rnnoise_process_frame(state, output, last_samples);
        (void)vad;
        if (push_worker_result(context, output, previous_vad, 1, 0,
                               input_frame.input_samples_total) != 0) {
          shared_fail(context->shared, "writer queue closed");
          goto done;
        }
      }
      if (push_worker_result(context, NULL, 0, 0, 1,
                             input_frame.input_samples_total) != 0) {
        shared_fail(context->shared, "writer queue closed");
        goto done;
      }
      break;
    }
    vad = rnnoise_process_frame(state, output, input_frame.samples);
    if (have_previous) {
      if (push_worker_result(context, output, previous_vad, 1, 0, 0) != 0) {
        shared_fail(context->shared, "writer queue closed");
        goto done;
      }
    }
    memcpy(last_samples, input_frame.samples, sizeof(last_samples));
    previous_vad = vad;
    have_previous = 1;
  }
  if (!input_frame.is_end && !shared_has_failed(context->shared)) {
    shared_fail(context->shared, "inference queue ended unexpectedly");
  }

done:
  rnnoise_destroy(state);
  queue_close(context->output_queue);
  return NULL;
}

static int write_wav_header(FILE *file, const AudioFormat *format) {
  unsigned char header[44];
  uint32_t block_align = (uint32_t)format->channels *
                         (uint32_t)sample_bytes(format->encoding);
  uint16_t wav_format = sample_encoding_is_float(format->encoding) ? 3 : 1;
  uint32_t byte_rate = format->sample_rate * block_align;
  memset(header, 0, sizeof(header));
  memcpy(header, "RIFF", 4);
  header[4] = 0xff;
  header[5] = 0xff;
  header[6] = 0xff;
  header[7] = 0xff;
  memcpy(header + 8, "WAVEfmt ", 8);
  header[16] = 16;
  header[20] = (unsigned char)(wav_format & 0xff);
  header[21] = (unsigned char)(wav_format >> 8);
  header[22] = (unsigned char)(format->channels & 0xff);
  header[23] = (unsigned char)(format->channels >> 8);
  header[24] = (unsigned char)(format->sample_rate & 0xff);
  header[25] = (unsigned char)((format->sample_rate >> 8) & 0xff);
  header[26] = (unsigned char)((format->sample_rate >> 16) & 0xff);
  header[27] = (unsigned char)((format->sample_rate >> 24) & 0xff);
  header[28] = (unsigned char)(byte_rate & 0xff);
  header[29] = (unsigned char)((byte_rate >> 8) & 0xff);
  header[30] = (unsigned char)((byte_rate >> 16) & 0xff);
  header[31] = (unsigned char)((byte_rate >> 24) & 0xff);
  header[32] = (unsigned char)(block_align & 0xff);
  header[33] = (unsigned char)(block_align >> 8);
  header[34] = (unsigned char)(block_align * 8 / format->channels);
  header[35] = (unsigned char)((block_align * 8 / format->channels) >> 8);
  memcpy(header + 36, "data", 4);
  header[40] = 0xff;
  header[41] = 0xff;
  header[42] = 0xff;
  header[43] = 0xff;
  return fwrite(header, 1, sizeof(header), file) == sizeof(header) ? 0 : -1;
}

static int finalize_wav(FILE *file) {
  uint64_t end;
  uint64_t data_size;
  unsigned char value[4];
  if (fflush(file) != 0) return -1;
  if (fseeko(file, 0, SEEK_END) != 0) return 0;
  {
    off_t position = ftello(file);
    if (position < 0) return 0;
    end = (uint64_t)position;
  }
  if (end < 44) return -1;
  data_size = end - 44;
  if (data_size > UINT32_MAX - 36) return -1;
  value[0] = (unsigned char)((data_size + 36) & 0xff);
  value[1] = (unsigned char)(((data_size + 36) >> 8) & 0xff);
  value[2] = (unsigned char)(((data_size + 36) >> 16) & 0xff);
  value[3] = (unsigned char)(((data_size + 36) >> 24) & 0xff);
  if (fseeko(file, 4, SEEK_SET) != 0 ||
      fwrite(value, 1, 4, file) != 4) return -1;
  value[0] = (unsigned char)(data_size & 0xff);
  value[1] = (unsigned char)((data_size >> 8) & 0xff);
  value[2] = (unsigned char)((data_size >> 16) & 0xff);
  value[3] = (unsigned char)((data_size >> 24) & 0xff);
  if (fseeko(file, 40, SEEK_SET) != 0 ||
      fwrite(value, 1, 4, file) != 4) return -1;
  return fflush(file) == 0 ? 0 : -1;
}

static int writer_flush_samples(WriterContext *context, const float *samples,
                                size_t count) {
  size_t bytes_per_frame = (size_t)context->format.channels *
                           (size_t)sample_bytes(context->format.encoding);
  size_t offset = 0;
  while (offset < count) {
    size_t frames = (context->encode_capacity / bytes_per_frame);
    size_t sample;
    size_t channel;
    if (frames == 0) return -1;
    if (frames > count - offset) frames = count - offset;
    for (sample = 0; sample < frames; sample++) {
      for (channel = 0; channel < (size_t)context->format.channels; channel++) {
        encode_sample(samples[offset + sample], context->format.encoding,
                      context->encode_buffer +
                      (sample * context->format.channels + channel) *
                      sample_bytes(context->format.encoding));
      }
    }
    if (fwrite(context->encode_buffer, bytes_per_frame, frames,
               context->audio_file) != frames)
      return -1;
    offset += frames;
  }
  context->written_frames += count;
  return 0;
}

static int writer_tail_add(WriterContext *context, const float *samples,
                           size_t count) {
  if (count >= context->hold_capacity) {
    size_t keep = context->hold_capacity;
    if (context->hold_length > 0 &&
        writer_flush_samples(context, context->hold,
                             context->hold_length) != 0)
      return -1;
    if (writer_flush_samples(context, samples, count - keep) != 0)
      return -1;
    memcpy(context->hold, samples + count - keep, keep * sizeof(float));
    context->hold_length = keep;
    return 0;
  }
  if (context->hold_length + count > context->hold_capacity) {
    size_t excess = context->hold_length + count - context->hold_capacity;
    if (writer_flush_samples(context, context->hold, excess) != 0)
      return -1;
    memmove(context->hold, context->hold + excess,
            (context->hold_length - excess) * sizeof(float));
    context->hold_length -= excess;
  }
  memcpy(context->hold + context->hold_length, samples,
         count * sizeof(float));
  context->hold_length += count;
  return 0;
}

static int writer_pull_resampler(WriterContext *context) {
  while (resampler_can_read(&context->resampler)) {
    size_t count = 0;
    while (count < context->chunk_capacity &&
           resampler_can_read(&context->resampler)) {
      context->chunk[count++] = resampler_read(&context->resampler);
    }
    if (writer_tail_add(context, context->chunk, count) != 0) return -1;
  }
  return 0;
}

static int writer_process_frame(WriterContext *context, const Frame *frame) {
  if (context->bypass_resampler) {
    if (writer_tail_add(context, frame->samples, RNNOISE_FRAME) != 0)
      return -1;
  } else {
    if (resampler_feed(&context->resampler, frame->samples,
                       RNNOISE_FRAME) != 0)
      return -1;
    context->clean_samples += RNNOISE_FRAME;
    if (writer_pull_resampler(context) != 0) return -1;
  }
  if (context->bypass_resampler)
    context->clean_samples += RNNOISE_FRAME;
  if (frame->has_vad) {
    if (fprintf(context->vad_file, "%.6f,%.9f\n",
                (double)context->vad_rows / 100.0,
                (double)frame->vad) < 0)
      return -1;
    context->vad_rows++;
  }
  return 0;
}

static int writer_finish(WriterContext *context, uint64_t source_frames) {
  uint64_t remaining;
  if (!context->bypass_resampler) {
    resampler_finish(&context->resampler, context->clean_samples);
    while (context->resampler.total_out < context->resampler.target_out) {
      size_t count = 0;
      while (count < context->chunk_capacity &&
             context->resampler.total_out <
             context->resampler.target_out)
        context->chunk[count++] = resampler_read(&context->resampler);
      if (writer_tail_add(context, context->chunk, count) != 0) return -1;
    }
  }
  if (context->written_frames > source_frames) return -1;
  remaining = source_frames - context->written_frames;
  if (remaining > context->hold_length) return -1;
  if (remaining > 0 &&
      writer_flush_samples(context, context->hold, (size_t)remaining) != 0)
    return -1;
  if (context->output_wav && finalize_wav(context->audio_file) != 0)
    return -1;
  if (fflush(context->vad_file) != 0) return -1;
  return 0;
}

static void *writer_thread(void *argument) {
  WriterContext *context = (WriterContext *)argument;
  Frame frame;
  uint64_t source_frames = 0;
  if (context->output_wav &&
      write_wav_header(context->audio_file, &context->format) != 0) {
    shared_fail(context->shared, "failed to write WAV header");
    goto fail;
  }
  if (fprintf(context->vad_file, "time_seconds,vad_probability\n") < 0) {
    shared_fail(context->shared, "failed to write VAD CSV header");
    goto fail;
  }
  while (queue_pop(context->input_queue, &frame) == 0) {
    if (frame.is_end) {
      source_frames = frame.input_samples_total;
      break;
    }
    if (writer_process_frame(context, &frame) != 0) {
      shared_fail(context->shared, "failed to write output");
      goto fail;
    }
  }
  if (!frame.is_end) {
    shared_fail(context->shared, "writer queue ended unexpectedly");
    goto fail;
  }
  if (writer_finish(context, source_frames) != 0) {
    shared_fail(context->shared,
                "failed to finalize output (source/output length mismatch)");
    goto fail;
  }
  return NULL;

fail:
  queue_close(context->input_queue);
  return NULL;
}

static void usage(FILE *file) {
  fprintf(file,
    "Usage: rnnoise_stream [options] INPUT OUTPUT\n"
    "\n"
    "Stream WAV or raw PCM through RNNoise at 10 ms per inference frame.\n"
    "Use '-' for stdin/stdout. VAD is written as time_seconds,probability.\n"
    "\n"
    "Options:\n"
    "  -f, --input-format auto|wav|raw     Input format (default: auto)\n"
    "  -F, --output-format auto|wav|raw    Output format (default: auto)\n"
    "  -r, --sample-rate HZ                Raw input rate (default: 48000)\n"
    "  -c, --channels N                    Raw input channels (default: 1)\n"
    "      --sample-format FORMAT          Raw input format (default: s16le)\n"
    "      --output-sample-format FORMAT   Raw output format (default: input)\n"
    "      --vad FILE                      VAD CSV path (default: derived)\n"
    "      --model FILE                    Load a custom RNNoise model\n"
    "  -q, --queue-frames N                Bounded queue size (default: 32)\n"
    "      --quiet                         Suppress the startup summary\n"
    "  -h, --help                          Show this help\n"
    "\n"
    "Formats: u8, s16le, s16be, s24le, s24be, s32le, s32be, f32le, f32be.\n"
    "For raw stdin, use --sample-format and --sample-rate as needed.\n");
}

static int output_format_is_wav(const Config *config, int input_is_wav) {
  const char *extension;
  if (strcmp(config->output_format, "wav") == 0) return 1;
  if (strcmp(config->output_format, "raw") == 0) return 0;
  if (strcmp(config->output_path, "-") == 0) return input_is_wav;
  extension = strrchr(config->output_path, '.');
  return extension != NULL && strcasecmp(extension, ".wav") == 0;
}

static char *derive_vad_path(const Config *config) {
  const char *base;
  size_t length;
  char *path;
  if (config->vad_path != NULL) {
    length = strlen(config->vad_path);
    path = (char *)malloc(length + 1);
    if (path != NULL) memcpy(path, config->vad_path, length + 1);
    return path;
  }
  if (strcmp(config->output_path, "-") != 0)
    base = config->output_path;
  else if (strcmp(config->input_path, "-") != 0)
    base = config->input_path;
  else
    return NULL;
  length = strlen(base) + strlen(".vad.csv");
  path = (char *)malloc(length + 1);
  if (path == NULL) return NULL;
  snprintf(path, length + 1, "%s.vad.csv", base);
  return path;
}

static int parse_options(int argc, char **argv, Config *config) {
  static const struct option options[] = {
    {"input-format", required_argument, NULL, 'f'},
    {"output-format", required_argument, NULL, 'F'},
    {"sample-rate", required_argument, NULL, 'r'},
    {"channels", required_argument, NULL, 'c'},
    {"sample-format", required_argument, NULL, 1000},
    {"output-sample-format", required_argument, NULL, 1001},
    {"vad", required_argument, NULL, 1002},
    {"model", required_argument, NULL, 1003},
    {"queue-frames", required_argument, NULL, 'q'},
    {"quiet", no_argument, NULL, 'Q'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
  };
  int option;
  memset(config, 0, sizeof(*config));
  config->input_format = "auto";
  config->output_format = "auto";
  config->sample_format = "s16le";
  config->raw_rate = RNNOISE_RATE;
  config->raw_channels = 1;
  config->queue_frames = DEFAULT_QUEUE_FRAMES;
  while ((option = getopt_long(argc, argv, "f:F:r:c:q:Qh", options,
                               NULL)) != -1) {
    switch (option) {
      case 'f': config->input_format = optarg; break;
      case 'F': config->output_format = optarg; break;
      case 'r': {
        char *end;
        unsigned long value = strtoul(optarg, &end, 10);
        if (*optarg == '\0' || *end != '\0' || value == 0 ||
            value > MAX_RATE) return -1;
        config->raw_rate = (uint32_t)value;
        break;
      }
      case 'c': {
        char *end;
        long value = strtol(optarg, &end, 10);
        if (*optarg == '\0' || *end != '\0' || value < 1 ||
            value > MAX_CHANNELS) return -1;
        config->raw_channels = (int)value;
        break;
      }
      case 'q': {
        char *end;
        long value = strtol(optarg, &end, 10);
        if (*optarg == '\0' || *end != '\0' || value < 2 || value > 4096)
          return -1;
        config->queue_frames = (int)value;
        break;
      }
      case 1000: config->sample_format = optarg; break;
      case 1001: config->output_sample_format = optarg; break;
      case 1002: config->vad_path = optarg; break;
      case 1003: config->model_path = optarg; break;
      case 'Q': config->quiet = 1; break;
      case 'h': usage(stdout); exit(0);
      default: return -1;
    }
  }
  if (argc - optind != 2) return -1;
  config->input_path = argv[optind];
  config->output_path = argv[optind + 1];
  if (strcmp(config->input_format, "auto") != 0 &&
      strcmp(config->input_format, "wav") != 0 &&
      strcmp(config->input_format, "raw") != 0)
    return -1;
  if (strcmp(config->output_format, "auto") != 0 &&
      strcmp(config->output_format, "wav") != 0 &&
      strcmp(config->output_format, "raw") != 0)
    return -1;
  return 0;
}

static int writer_context_init(WriterContext *context, FrameQueue *queue,
                               SharedState *shared, FILE *audio_file,
                               FILE *vad_file, const AudioFormat *format,
                               int output_wav) {
  uint64_t rate;
  memset(context, 0, sizeof(*context));
  context->input_queue = queue;
  context->shared = shared;
  context->audio_file = audio_file;
  context->vad_file = vad_file;
  context->format = *format;
  context->output_wav = output_wav;
  context->bypass_resampler = format->sample_rate == RNNOISE_RATE;
  if (!context->bypass_resampler)
    resampler_init(&context->resampler, RNNOISE_RATE, format->sample_rate);
  rate = format->sample_rate;
  context->chunk_capacity = (size_t)(rate / 100 + 64);
  context->hold_capacity = context->chunk_capacity * 2 + 64;
  context->chunk = (float *)malloc(context->chunk_capacity * sizeof(float));
  context->hold = (float *)malloc(context->hold_capacity * sizeof(float));
  context->encode_capacity = 65536;
  context->encode_buffer = (unsigned char *)malloc(context->encode_capacity);
  if (context->chunk == NULL || context->hold == NULL ||
      context->encode_buffer == NULL)
    return -1;
  return 0;
}

static void writer_context_destroy(WriterContext *context) {
  free(context->chunk);
  free(context->hold);
  free(context->encode_buffer);
}

int main(int argc, char **argv) {
  Config config;
  AudioInput input;
  AudioFormat output_format;
  SampleEncoding output_encoding;
  FILE *output_file = NULL;
  FILE *vad_file = NULL;
  char *vad_path = NULL;
  FrameQueue input_queue;
  FrameQueue output_queue;
  SharedState shared;
  ReaderContext reader_context;
  WorkerContext worker_context;
  WriterContext writer_context;
  pthread_t reader;
  pthread_t worker;
  pthread_t writer;
  int reader_started = 0;
  int worker_started = 0;
  int writer_started = 0;
  int queues_initialized = 0;
  int input_opened = 0;
  int output_wav;
  int result = 1;
  char error[512];
  RNNModel *model = NULL;
  memset(&input, 0, sizeof(input));
  memset(&shared, 0, sizeof(shared));
  memset(&reader_context, 0, sizeof(reader_context));
  memset(&worker_context, 0, sizeof(worker_context));
  memset(&writer_context, 0, sizeof(writer_context));
  if (parse_options(argc, argv, &config) != 0) {
    usage(stderr);
    return 1;
  }
  if (open_audio_input(&config, &input, error, sizeof(error)) != 0) {
    fprintf(stderr, "rnnoise_stream: %s\n", error);
    close_audio_input(&input);
    goto cleanup;
  }
  input_opened = 1;
  output_encoding = input.format.encoding;
  if (config.output_sample_format != NULL &&
      parse_sample_format(config.output_sample_format,
                          &output_encoding) != 0) {
    fprintf(stderr, "rnnoise_stream: unknown output sample format '%s'\n",
            config.output_sample_format);
    goto cleanup;
  }
  output_format.sample_rate = input.format.sample_rate;
  output_format.channels = input.format.channels;
  output_format.encoding = output_encoding;
  output_wav = output_format_is_wav(&config, input.is_wav);
  if (output_wav)
    output_format.encoding = little_endian_encoding(output_format.encoding);
  if (strcmp(config.output_path, "-") == 0) {
    output_file = stdout;
  } else {
    if (strcmp(config.input_path, config.output_path) == 0) {
      fprintf(stderr, "rnnoise_stream: input and output paths must differ\n");
      goto cleanup;
    }
    output_file = fopen(config.output_path, "wb");
    if (output_file == NULL) {
      fprintf(stderr, "rnnoise_stream: cannot open '%s': %s\n",
              config.output_path, strerror(errno));
      goto cleanup;
    }
  }
  vad_path = derive_vad_path(&config);
  if (vad_path == NULL) {
    fprintf(stderr,
            "rnnoise_stream: --vad is required when both audio streams are "
            "stdin/stdout\n");
    goto cleanup;
  }
  if (strcmp(vad_path, "-") == 0) {
    if (output_file == stdout) {
      fprintf(stderr,
              "rnnoise_stream: VAD CSV and audio cannot both use stdout\n");
      goto cleanup;
    }
    vad_file = stdout;
  } else {
    if (strcmp(vad_path, config.output_path) == 0) {
      fprintf(stderr, "rnnoise_stream: VAD and audio paths must differ\n");
      goto cleanup;
    }
    if (strcmp(vad_path, config.input_path) == 0) {
      fprintf(stderr, "rnnoise_stream: VAD and input paths must differ\n");
      goto cleanup;
    }
    vad_file = fopen(vad_path, "wb");
    if (vad_file == NULL) {
      fprintf(stderr, "rnnoise_stream: cannot open '%s': %s\n",
              vad_path, strerror(errno));
      goto cleanup;
    }
  }
  if (pthread_mutex_init(&shared.mutex, NULL) != 0) {
    fprintf(stderr, "rnnoise_stream: failed to initialize mutex\n");
    goto cleanup;
  }
  if (queue_init(&input_queue, config.queue_frames) != 0) {
    fprintf(stderr, "rnnoise_stream: failed to initialize input queue\n");
    pthread_mutex_destroy(&shared.mutex);
    goto cleanup;
  }
  if (queue_init(&output_queue, config.queue_frames) != 0) {
    fprintf(stderr, "rnnoise_stream: failed to initialize output queue\n");
    queue_destroy(&input_queue);
    pthread_mutex_destroy(&shared.mutex);
    goto cleanup;
  }
  queues_initialized = 1;
  if (config.model_path != NULL) {
    model = rnnoise_model_from_filename(config.model_path);
    if (model == NULL) {
      fprintf(stderr, "rnnoise_stream: cannot load model '%s'\n",
              config.model_path);
      goto cleanup;
    }
  }
  if (writer_context_init(&writer_context, &output_queue, &shared,
                          output_file, vad_file, &output_format,
                          output_wav) != 0) {
    fprintf(stderr, "rnnoise_stream: out of memory\n");
    goto cleanup;
  }
  if (!config.quiet) {
    fprintf(stderr,
            "rnnoise_stream: %s %u Hz, %d ch -> %s %u Hz, %d ch, "
            "%d-frame queues\n",
            input.is_wav ? "WAV" : "raw", input.format.sample_rate,
            input.format.channels, output_wav ? "WAV" : "raw",
            output_format.sample_rate, output_format.channels,
            config.queue_frames);
  }
  reader_context.input = &input;
  reader_context.output_queue = &input_queue;
  reader_context.shared = &shared;
  worker_context.input_queue = &input_queue;
  worker_context.output_queue = &output_queue;
  worker_context.shared = &shared;
  worker_context.model = model;
  if (pthread_create(&writer, NULL, writer_thread, &writer_context) != 0) {
    fprintf(stderr, "rnnoise_stream: failed to start writer thread\n");
    goto cleanup;
  }
  writer_started = 1;
  if (pthread_create(&worker, NULL, worker_thread, &worker_context) != 0) {
    fprintf(stderr, "rnnoise_stream: failed to start inference thread\n");
    queue_close(&input_queue);
    queue_close(&output_queue);
    goto join_threads;
  }
  worker_started = 1;
  if (pthread_create(&reader, NULL, reader_thread, &reader_context) != 0) {
    fprintf(stderr, "rnnoise_stream: failed to start reader thread\n");
    queue_close(&input_queue);
    queue_close(&output_queue);
    goto join_threads;
  }
  reader_started = 1;

join_threads:
  if (reader_started) pthread_join(reader, NULL);
  if (worker_started) pthread_join(worker, NULL);
  if (writer_started) pthread_join(writer, NULL);
  if (!reader_started || !worker_started || !writer_started)
    shared.failed = 1;
  result = shared.failed ? 1 : 0;

cleanup:
  if (queues_initialized) {
    queue_destroy(&input_queue);
    queue_destroy(&output_queue);
    pthread_mutex_destroy(&shared.mutex);
  }
  if (output_file != NULL && output_file != stdout) fclose(output_file);
  else if (output_file == stdout) fflush(stdout);
  if (vad_file != NULL && vad_file != stdout) fclose(vad_file);
  else if (vad_file == stdout) fflush(stdout);
  writer_context_destroy(&writer_context);
  if (model != NULL) rnnoise_model_free(model);
  if (input_opened) close_audio_input(&input);
  free(vad_path);
  return result;
}
