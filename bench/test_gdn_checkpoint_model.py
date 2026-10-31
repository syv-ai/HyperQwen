#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real TP=2 GDN export/replay probe; no performance or cache-capacity claim.

Run --variant enabled then default-off with the same --output-dir, each in a
fresh process. Add -dflash to both variants for the actual seven-draft path.
--compare-only checks saved prompt hashes/output tokens without loading a model.
--require-off-miss additionally refuses a result whose baseline already hits.
Needs two free CUDA GPUs, cached model weights and the patched current vLLM.
"""
import argparse
import hashlib
import json
import os
import types
from collections.abc import Mapping
from pathlib import Path



class GDNProbe:
    def checkpoint_probe_install(self):
        model = self.model_runner.get_model()
        found = [(name, mod) for name, mod in model.named_modules()
                 if hasattr(mod, '_store_gdn_prefill_checkpoint')]
        if not found:
            raise RuntimeError('No GDN exporter found')
        name, layer = found[0]
        self._cp_records = []
        self._cp_forwards = []
        self._cp_drafts = 0
        original = layer._store_gdn_prefill_checkpoint
        records = self._cp_records
        forwards = self._cp_forwards

        def observe(this, metadata, *args, **kwargs):
            forwards.append(args[1].shape[1])
            result = original(metadata, *args, **kwargs)
            plan = metadata.checkpoint_plan
            if plan is not None:
                records.append({'slices': plan.source_slices,
                                'destinations': plan.state_indices.cpu().tolist()})
            return result

        layer._store_gdn_prefill_checkpoint = types.MethodType(observe, layer)
        speculator = self.model_runner.speculator
        if speculator is not None:
            propose = speculator.propose

            def observe_draft(this, *args, **kwargs):
                self._cp_drafts += 1
                return propose(*args, **kwargs)

            speculator.propose = types.MethodType(observe_draft, speculator)
        sizes = [g.kv_cache_spec.block_size for g in self.model_runner.kv_cache_config.kv_cache_groups]
        return {'layer': name, 'backend': layer.gdn_prefill_backend,
                'hash_block_size': self.vllm_config.cache_config.prefix_match_unit or min(sizes),
                'speculator': type(speculator).__name__}

    def checkpoint_probe_read(self):
        return list(self._cp_records)

    def checkpoint_probe_forwards(self):
        return list(self._cp_forwards)

    def checkpoint_probe_drafting(self):
        return self._cp_drafts

    def checkpoint_probe_reset(self):
        self._cp_records.clear()
        self._cp_forwards.clear()
        self._cp_drafts = 0
        return True


def compare_results(directory, dflash, require_off_miss):
    suffix = '-dflash' if dflash else ''
    enabled = json.loads((directory / f'gdn-model-enabled{suffix}.json').read_text())
    baseline = json.loads((directory / f'gdn-model-default-off{suffix}.json').read_text())
    assert enabled.get('passed') and baseline.get('passed')
    for a, b in zip(enabled['records'], baseline['records']):
        assert a['prompt_sha256'] == b['prompt_sha256']
        assert a['token_ids'] == b['token_ids'], (a, b)
    on_hit = enabled['records'][2]['cached_tokens']
    off_hit = baseline['records'][2]['cached_tokens']
    print(json.dumps({'outputs_identical': True, 'enabled_hit': on_hit, 'off_hit': off_hit,
                      'enabled_only_hit': on_hit > 0 and off_hit == 0}))
    if require_off_miss:
        assert on_hit > 0 and off_hit == 0, 'requested enabled-hit/default-off-miss criterion was not observed'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=('enabled', 'default-off', 'enabled-dflash', 'default-off-dflash'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='/app/models/Qwen3.8-27B-W4A16-AutoRound')
    parser.add_argument('--drafter', default='/app/models/Qwen3.8-27B-DFlash2-W4A16')
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--require-off-miss', action='store_true')
    args = parser.parse_args()
    if args.compare_only:
        compare_results(args.output_dir, args.variant.endswith('dflash'), args.require_off_miss)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant = args.variant
    os.environ['VLLM_GDN_PREFILL_CHECKPOINTS'] = '1' if variant.startswith('enabled') else '0'
    from vllm import LLM, SamplingParams
    enabled = variant.startswith('enabled')
    speculative = variant.endswith('dflash')
    result = {'variant': variant, 'records': []}
    extra = {}
    if speculative:
        extra['speculative_config'] = {
            'method': 'dflash', 'num_speculative_tokens': 7,
            'model': args.drafter}
    llm = LLM(model=args.model, tensor_parallel_size=2,
              dtype='bfloat16', max_model_len=11264, max_num_seqs=2,
              max_num_batched_tokens=8192, gpu_memory_utilization=.80,
              kv_cache_memory_bytes=1500000000,
              kv_cache_dtype='auto', attention_backend='FLASH_ATTN',
              enforce_eager=True, enable_prefix_caching=True, language_model_only=True,
              mamba_cache_mode='align', seed=42,
              worker_extension_cls='test_gdn_checkpoint_model.GDNProbe', **extra)
    try:
        result['layers'] = llm.collective_rpc('checkpoint_probe_install')
        tokenizer = llm.get_tokenizer()
        material = 'The unique archive passcode is 483729.\n' + ''.join(
            f'Archive entry {i}: the blue cedar tree stands beside a quiet river.\n'
            for i in range(2000))
        tokens = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': material}], tokenize=True,
            add_generation_prompt=False)
        if isinstance(tokens, Mapping):
            tokens = tokens['input_ids']
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        print('PROMPT_LENGTH', len(tokens), flush=True)
        prefix = tokens[:8000]
        assert len(prefix) == 8000
        tail = tokenizer.encode('\nWhat is the unique archive passcode? Reply with digits only.'
                                '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
                                add_special_tokens=False)
        params = SamplingParams(temperature=0, max_tokens=12)

        def generate(label, ids):
            llm.collective_rpc('checkpoint_probe_reset')
            out = llm.generate({'prompt_token_ids': ids}, params, use_tqdm=False)[0]
            row = {'label': label, 'prompt_tokens': len(ids),
                   'prompt_sha256': hashlib.sha256(bytes(str(ids), 'utf8')).hexdigest(),
                   'cached_tokens': out.num_cached_tokens, 'text': out.outputs[0].text,
                   'token_ids': out.outputs[0].token_ids,
                   'checkpoints': llm.collective_rpc('checkpoint_probe_read'),
                   'prefill_forwards': llm.collective_rpc('checkpoint_probe_forwards'),
                   'draft_calls': llm.collective_rpc('checkpoint_probe_drafting')}
            result['records'].append(row)
            print('RESULT', json.dumps(row), flush=True)
            return row

        cold = generate('cold', prefix + tail)
        # The actual allocator-selected interior offset is recorded on the enabled
        # run. On the default-off baseline use the same persisted prompt plan.
        plan_file = args.output_dir / ('gdn-prefix-plan' + ('-dflash' if speculative else '') + '.json')
        if enabled:
            offsets = [start + length for records in cold['checkpoints']
                       for record in records for start, length in record['slices']]
            assert offsets, 'No exact checkpoint exported by the real model'
            offset = max(value for value in offsets if value < 8000)
            plan_file.write_text(json.dumps({'offset': offset}) + '\n')
        else:
            offset = json.loads(plan_file.read_text())['offset']
        generate('interleave', tokenizer.encode('Count from 1 to 3.', add_special_tokens=False))
        # Eagle-style matching proves one additional complete hash unit, then
        # drops it. Share that proof block too; it is not part of the resumed
        # recurrent state. Matching only through the checkpoint cannot hit it.
        shared = offset + (result['layers'][0]['hash_block_size'] if speculative else 0)
        assert shared <= len(prefix), (shared, offset, result['layers'])
        resumed = generate('interior-resume', prefix[:shared] + tail)
        repeated = generate('interior-repeat', prefix[:shared] + tail)
        # A long fresh request plus short traffic exercises intermediate chunks
        # and mixed prefill/decode scheduling when speculative decoding is on.
        if speculative and enabled:
            long_ids = [tokenizer.encode('Independent', add_special_tokens=False)[0]] + tokens[:11000] + tail
            llm.collective_rpc('checkpoint_probe_reset')
            outputs = llm.generate([{'prompt_token_ids': long_ids},
                                    {'prompt_token_ids': tokenizer.encode('Count from 1 to 3.', add_special_tokens=False)}],
                                   params, use_tqdm=False)
            result['mixed'] = {'outputs': [o.outputs[0].text for o in outputs],
                               'checkpoints': llm.collective_rpc('checkpoint_probe_read')}
            assert '483729' in outputs[0].outputs[0].text
        assert '483729' in cold['text'], cold
        assert '483729' in resumed['text'], resumed
        assert resumed['token_ids'] == repeated['token_ids']
        if enabled:
            assert resumed['cached_tokens'] >= offset, resumed
            assert all(cold['checkpoints']), 'Every TP rank must export states'
        if speculative:
            assert all(cold['draft_calls']), 'DFlash propose must actually execute'
        result['checkpoint_offset'] = offset
        result['passed'] = True
    finally:
        (args.output_dir / f'gdn-model-{variant}.json').write_text(json.dumps(result, indent=2) + '\n')
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
