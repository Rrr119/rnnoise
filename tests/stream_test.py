#!/usr/bin/env python3
"""End-to-end checks for rnnoise_stream.

The test synthesizes audio so it needs no external sample corpus.  It checks
bounded-queue processing, VAD timestamps, sample-accurate output length,
resampling, stdin/stdout pipes, and a real reduction in stationary noise.
"""

import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import wave


def wav_path(directory, name):
    return os.path.join(directory, name)


def write_wav(path, rate, samples):
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack("<%dh" % len(samples), *samples))


def read_wav(path):
    with wave.open(path, "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.getnframes()
        data = source.readframes(frames)
    if width != 2:
        raise AssertionError("test expected 16-bit output")
    samples = list(struct.unpack("<%dh" % (len(data) // 2), data))
    if channels != 1:
        samples = samples[::channels]
    return rate, channels, samples


def synth_noisy_sine(rate, duration=3.2):
    rng = random.Random(7420)
    total = int(rate * duration)
    silence_end = int(rate * 0.20)
    noise_end = int(rate * 1.20)
    tone_end = int(rate * 2.20)
    samples = []
    for index in range(total):
        noise = rng.uniform(-4500.0, 4500.0)
        if index < silence_end or index >= tone_end:
            value = 0.0
        elif index < noise_end:
            value = noise
        else:
            value = 11000.0 * math.sin(2.0 * math.pi * 440.0 * index / rate)
            value += noise
        samples.append(max(-32768, min(32767, int(round(value)))))
    return samples, silence_end, noise_end, tone_end


def rms(samples):
    if not samples:
        return 0.0
    return math.sqrt(sum(float(value) * value for value in samples) /
                     len(samples))


def read_vad(path):
    with open(path, "r", encoding="ascii") as source:
        lines = [line.strip() for line in source if line.strip()]
    if not lines or lines[0] != "time_seconds,vad_probability":
        raise AssertionError("bad VAD CSV header")
    rows = []
    for line in lines[1:]:
        timestamp, probability = line.split(",")
        rows.append((float(timestamp), float(probability)))
    return rows


def run_tool(tool, arguments, input_data=None):
    command = [tool] + arguments
    completed = subprocess.run(
        command,
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "command failed (%d): %s\n%s" %
            (completed.returncode, " ".join(command),
             completed.stderr.decode("utf-8", "replace")))
    return completed.stdout


def check_vad_alignment(rows, rate, output_frames):
    expected_rows = (output_frames * 100 + rate - 1) // rate
    if len(rows) != expected_rows:
        raise AssertionError(
            "VAD rows %d do not match audio frames %d" %
            (len(rows), expected_rows))
    for index, (timestamp, probability) in enumerate(rows):
        expected_time = index * 0.01
        if abs(timestamp - expected_time) > 1e-6:
            raise AssertionError(
                "VAD row %d has timestamp %.9f, expected %.9f" %
                (index, timestamp, expected_time))
        if probability < 0.0 or probability > 1.0:
            raise AssertionError("VAD probability out of range")


def test_48k_denoise_and_vad(tool, directory):
    input_path = wav_path(directory, "noisy.wav")
    output_path = wav_path(directory, "clean.wav")
    vad_path = wav_path(directory, "clean.vad.csv")
    rate = 48000
    samples, silence_end, noise_end, tone_end = synth_noisy_sine(rate)
    write_wav(input_path, rate, samples)
    run_tool(tool, [input_path, output_path, "--vad", vad_path,
                    "--queue-frames", "4", "--quiet"])
    output_rate, channels, output_samples = read_wav(output_path)
    if output_rate != rate or channels != 1:
        raise AssertionError("output format was not preserved")
    if len(output_samples) != len(samples):
        raise AssertionError(
            "output has %d frames, expected %d" %
            (len(output_samples), len(samples)))
    noise_start = int(rate * 0.55)
    input_noise = rms(samples[noise_start:noise_end])
    output_noise = rms(output_samples[noise_start:noise_end])
    if output_noise >= input_noise * 0.75:
        raise AssertionError(
            "noise was not reduced enough: input RMS %.1f, output RMS %.1f" %
            (input_noise, output_noise))
    silence_start = int(rate * 2.55)
    silence_vad = [row[1] for row in read_vad(vad_path)
                   if row[0] >= silence_start / float(rate)]
    if not silence_vad or sum(silence_vad) / len(silence_vad) > 0.02:
        raise AssertionError("VAD did not stay near zero in silence")
    rows = read_vad(vad_path)
    check_vad_alignment(rows, rate, len(output_samples))


def test_44100_resampling(tool, directory):
    input_path = wav_path(directory, "input-44100.wav")
    output_path = wav_path(directory, "output-44100.wav")
    vad_path = wav_path(directory, "output-44100.vad.csv")
    rate = 44100
    samples, _, _, _ = synth_noisy_sine(rate, duration=1.003)
    write_wav(input_path, rate, samples)
    run_tool(tool, [input_path, output_path, "--vad", vad_path, "--quiet"])
    output_rate, channels, output_samples = read_wav(output_path)
    if output_rate != rate or channels != 1:
        raise AssertionError("resampled output did not preserve sample rate")
    if len(output_samples) != len(samples):
        raise AssertionError(
            "resampled output has %d frames, expected %d" %
            (len(output_samples), len(samples)))
    check_vad_alignment(read_vad(vad_path), rate, len(output_samples))


def test_raw_stdin_stdout(tool, directory):
    rate = 48000
    samples, _, _, _ = synth_noisy_sine(rate, duration=0.30)
    raw = struct.pack("<%dh" % len(samples), *samples)
    vad_path = wav_path(directory, "pipe.vad.csv")
    output = run_tool(
        tool,
        ["-", "-", "--input-format", "raw", "--output-format", "raw",
         "--sample-format", "s16le", "--vad", vad_path, "--quiet"],
        input_data=raw,
    )
    if len(output) != len(raw):
        raise AssertionError(
            "pipe output has %d bytes, expected %d" %
            (len(output), len(raw)))
    check_vad_alignment(read_vad(vad_path), rate, len(output) // 2)


def main():
    tool = os.environ.get("RNNOISE_STREAM", "./rnnoise_stream")
    if not os.path.isfile(tool) or not os.access(tool, os.X_OK):
        raise SystemExit("rnnoise_stream executable not found: %s" % tool)
    with tempfile.TemporaryDirectory(prefix="rnnoise-stream-test-") as directory:
        test_48k_denoise_and_vad(tool, directory)
        test_44100_resampling(tool, directory)
        test_raw_stdin_stdout(tool, directory)
    print("rnnoise_stream tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
