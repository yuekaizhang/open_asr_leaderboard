import argparse
import os
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np

# from vad.pyannote import Pyannote
from whisper_utils import log_mel_spectrogram, get_tokenizer
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
wer_metric = evaluate.load("wer")

try:
    from tn.chinese.normalizer import Normalizer
except ImportError:
    print("Please pip install WeTextProcessing")
    raise

class FasterWhisper(object):

    def __init__(self,
                 model_id,
                 assets_dir="assets",
                 batch_size=64,
                 text_prefix="<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>",
                 n_mels=128):
        tokenizer_name = "multilingual"
        assert (Path(assets_dir) / "multilingual.tiktoken").exists(
        ), "multilingual.tiktoken file is not existed in assets_dir"

        self.tokenizer = get_tokenizer(name=tokenizer_name,
                                       num_languages=100,
                                       tokenizer_dir=assets_dir)
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special=self.tokenizer.special_tokens_set)[0]
        self.n_mels = n_mels
        self.text_prefix = text_prefix
        model = WhisperModel(model_id, device="cuda", compute_type="float16")
        self.model = BatchedInferencePipeline(model=model)
  
    def process_batch(self, mel, mel_input_lengths, num_threads=1, max_new_tokens=96):
        prompt_id = self.tokenizer.encode(
            self.text_prefix, allowed_special=self.tokenizer.special_tokens_set)
        prompt_id = torch.tensor(prompt_id).tolist()
        batch_size = len(mel)

        # if isinstance(mel, list):
        #     mel = torch.stack([m.transpose(1, 2).type(torch.float16).squeeze(0) for m in mel])
        # else:
        #     mel = mel.transpose(1, 2)
        if isinstance(mel, list):
            mel = torch.stack([m.type(torch.float16).squeeze(0) for m in mel])
        else:
            mel = mel
        encoder_output = self.model.model.encode(mel)
        results = self.model.model.model.generate(
            encoder_output,
            [prompt_id] * batch_size,
            beam_size=1,
            patience=1,
            length_penalty=1,
            max_length=448,
            return_scores=True,
            return_no_speech_prob=True,
        )
        texts = []
        for i, result in enumerate(results):
            output_ids = result.sequences_ids[0]
            text = self.tokenizer.decode(output_ids).strip()
            text = re.sub(r'<\|.*?\|>', '', text)
            texts.append(text)
        return texts

def main(args):
    asr_model = FasterWhisper(model_id=args.model_id)

    default_vad_options = {
        "chunk_size": 30,
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }
    # vad model copied from https://github.com/m-bain/whisperX/pull/888
    # vad_model = Pyannote(torch.device('cuda'), use_auth_token="", **default_vad_options)
    def benchmark(batch, min_new_tokens=None):
        # Load audio inputs
        max_duration, sample_rate, max_batch_size = 30, 16000, 64
        default_vad_options = {
            "chunk_size": 30,
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }
        audios_origin = [audio["array"].astype(np.float32) for audio in batch["audio"]]
        if "audio_length_s" not in batch:
            batch["audio_length_s"] = [len(audio) / sample_rate for audio in audios_origin]
        minibatch_size = len(audios_origin)
        audios, audio_index = [], []
        for i, audio in enumerate(audios_origin):
            if len(audio) > max_duration * sample_rate:
                waveform = vad_model.preprocess_audio(audio)
                vad_segments = vad_model({"waveform": waveform, "sample_rate": sample_rate})
                vad_segments = vad_model.merge_chunks(
                    vad_segments,
                    default_vad_options["chunk_size"],
                    onset=default_vad_options["vad_onset"],
                    offset=default_vad_options["vad_offset"]
                )
                for seg in vad_segments:
                    f1 = int(seg['start'] * sample_rate)
                    f2 = int(seg['end'] * sample_rate)
                    audios.append(audio[f1:f2])
                    audio_index.append(i)
            else:
                audios.append(audio)
                audio_index.append(i)
        audios = [torch.from_numpy(audio) for audio in audios]

        # START TIMING
        start_time = time.time()
        longest_duration = int(sample_rate * max_duration)

        features = [
            log_mel_spectrogram(wave,
                                asr_model.n_mels,
                                padding=longest_duration - wave.shape[-1],
                                device='cuda').unsqueeze(0)
            for wave in audios
        ]

        features_input_lengths = torch.tensor([f.shape[2] for f in features],
                                              dtype=torch.int32,
                                              device='cuda')

        texts_origin = asr_model.process_batch(features, features_input_lengths)

        # merge the transcriptions of the same audio
        texts = []
        for i in range(minibatch_size):
            text = []
            for j in range(len(texts_origin)):
                if audio_index[j] == i:
                    text.append(texts_origin[j])
            text = " ".join(text)
            texts.append(text)
        # END TIMING
        runtime = time.time() - start_time

        print(f"Batch size: {minibatch_size}, Time taken: {runtime:.2f} s, texts_origin_len: {len(texts_origin)}, texts_len: {len(texts)}")
        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in texts]
        zh_normalizer = Normalizer()
        # remove space and normalize Chinese transcriptions
        batch["norm_text"] = [ref.replace(" ", "") for ref in batch["norm_text"]]
        batch["norm_text"] = [zh_normalizer.normalize(ref) for ref in batch["norm_text"]]
        # add space back to the character level
        batch["norm_text"] = [" ".join(ref) for ref in batch["norm_text"]]
        batch["predictions"] = [zh_normalizer.normalize(pred) for pred in batch["predictions"]]
        batch["predictions"] = [" ".join(pred) for pred in batch["predictions"]]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        print(result)
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
