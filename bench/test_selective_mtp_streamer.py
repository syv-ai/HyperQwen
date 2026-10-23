"""CPU regressions for model-owned MTP safetensors shard selection.

Run after applying patches/series to the pinned vLLM runtime:
  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_selective_mtp_streamer.py
Real safetensors files and MTP consumer methods are used without a model/GPU.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from vllm.model_executor.model_loader import default_loader as default
from vllm.model_executor.model_loader import runai_streamer_loader as runai
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.models import qwen3_5_mtp as mtp


class SelectiveMTPShards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = [
            {
                "model.layers.0.weight_packed": torch.arange(6, dtype=torch.int32),
                "model.visual.blocks.0.weight": torch.ones(4, dtype=torch.bfloat16),
                "model.layers.0.mtp_like.weight": torch.zeros(2),
            },
            {
                "model.language_model.embed_tokens.weight_packed": torch.arange(
                    8, dtype=torch.int32
                ),
                "model.language_model.embed_tokens.weight_scale": torch.tensor(
                    [0.25], dtype=torch.bfloat16
                ),
                "lm_head.weight_packed": torch.arange(10, dtype=torch.int32),
                # A retained shard is still streamed in full.
                "model.layers.1.weight_packed": torch.ones(2, dtype=torch.int32),
            },
            {
                "mtp.layers.0.weight_packed": torch.arange(12, dtype=torch.int32),
                "mtp.layers.0.weight_scale": torch.tensor(
                    [0.125, 0.25], dtype=torch.bfloat16
                ),
                "mtp.norm.weight": torch.ones(4, dtype=torch.bfloat16),
            },
            {"mtp.draft_lm_head.weight_packed": torch.arange(5, dtype=torch.int32)},
        ]
        self.files = []
        for index, records in enumerate(self.records):
            path = self.root / f"{index}.safetensors"
            save_file(records, str(path))
            self.files.append(str(path))

    @staticmethod
    def model(enabled=True, cls=mtp.Qwen3_5MTP):
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model.model = SimpleNamespace(draft_lm_head=object() if enabled else None)
        return model

    @staticmethod
    def read(files, *args, **kwargs):
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    yield name, handle.get_tensor(name)

    @staticmethod
    def consume(model, weights):
        captured = {}

        class CaptureLoader:
            def __init__(self, owner):
                pass

            def load_weights(self, iterator):
                captured.update((name, tensor.clone()) for name, tensor in iterator)
                return set(captured)

        with patch.object(mtp, "AutoWeightsLoader", CaptureLoader):
            model.load_weights(weights)
        return captured

    def test_selection_reads_headers_only_and_preserves_file_order(self):
        def headers_only(*args, **kwargs):
            handle = safe_open(*args, **kwargs)
            return SimpleNamespaceContext(handle)

        with patch.object(weight_utils, "safe_open", headers_only):
            selected = weight_utils.filter_safetensors_files_by_weight_name(
                self.files, self.model().is_unused_checkpoint_weight
            )
        self.assertEqual(selected, self.files[1:])

    def test_rejected_checkpoint_keeps_original_list_for_model_errors(self):
        selected = weight_utils.filter_safetensors_files_by_weight_name(
            self.files, lambda name: True
        )
        self.assertIs(selected, self.files)

    def test_metadata_and_selection_errors_are_not_hidden(self):
        corrupt = self.root / "broken.safetensors"
        corrupt.write_bytes(b"not a checkpoint")
        with self.assertRaises(SafetensorError):
            weight_utils.filter_safetensors_files_by_weight_name(
                [str(corrupt)], lambda name: False
            )

        def broken_rule(name):
            raise ValueError("invalid model rule")

        with self.assertRaisesRegex(ValueError, "invalid model rule"):
            weight_utils.filter_safetensors_files_by_weight_name(
                self.files, broken_rule
            )

    def test_dense_and_moe_consumers_get_identical_tensor_names_dtypes_and_values(self):
        for cls in (mtp.Qwen3_5MTP, mtp.Qwen3_5MoeMTP):
            for enabled in (True, False):
                with self.subTest(model=cls.__name__, draft_head=enabled):
                    model = self.model(enabled, cls)
                    expected = self.consume(model, self.read(self.files))
                    selected = weight_utils.filter_safetensors_files_by_weight_name(
                        self.files, model.is_unused_checkpoint_weight
                    )
                    self.assertEqual(
                        selected, self.files[1:] if enabled else self.files[1:3]
                    )
                    actual = self.consume(model, self.read(selected))
                    self.assertEqual(actual.keys(), expected.keys())
                    self.assertIn("model.embed_tokens.weight_packed", actual)
                    self.assertIn("lm_head.weight_packed", actual)
                    self.assertIn("model.layers.0.weight_scale", actual)
                    for name, tensor in expected.items():
                        self.assertEqual(actual[name].dtype, tensor.dtype)
                        self.assertTrue(torch.equal(actual[name], tensor), name)

    def test_runai_uses_selected_shards_and_honors_model_weights_override(self):
        loader = runai.RunaiModelStreamerLoader.__new__(runai.RunaiModelStreamerLoader)
        loader._is_distributed = False
        loader.load_config = SimpleNamespace(use_tqdm_on_load=False)
        config = SimpleNamespace(
            model="unused/remote-model", model_weights=str(self.root), revision="pinned"
        )
        model = self.model()
        expected = self.consume(model, self.read(self.files))
        with (
            patch.object(
                loader, "_prepare_weights", return_value=self.files
            ) as prepare,
            patch.object(
                runai, "runai_safetensors_weights_iterator", wraps=self.read
            ) as stream,
            patch.object(mtp, "AutoWeightsLoader") as capture,
        ):
            received = {}
            capture.return_value.load_weights.side_effect = lambda weights: (
                received.update((name, tensor.clone()) for name, tensor in weights)
            )
            loader.load_weights(model, config)
        prepare.assert_called_once_with(str(self.root), "pinned")
        self.assertEqual(stream.call_args.args[0], self.files[1:])
        self.assertEqual(received.keys(), expected.keys())
        for name, tensor in expected.items():
            self.assertTrue(torch.equal(received[name], tensor), name)

    def test_runai_fallbacks_keep_streamer_file_list(self):
        for kind, distributed, path in [
            ("main", False, str(self.root)),
            ("namesake", False, str(self.root)),
            ("mtp", True, str(self.root)),
            ("mtp", False, "organization/model"),
            ("mtp", False, "s3://bucket/model"),
            ("mtp", False, "gs://bucket/model"),
            ("mtp", False, "az://container/model"),
        ]:
            with self.subTest(kind=kind, distributed=distributed, path=path):
                loader = runai.RunaiModelStreamerLoader.__new__(
                    runai.RunaiModelStreamerLoader
                )
                loader._is_distributed = distributed
                loader.load_config = SimpleNamespace(use_tqdm_on_load=False)
                model = SimpleNamespace(load_weights=lambda weights: list(weights))
                if kind == "mtp":
                    model.is_unused_checkpoint_weight = (
                        self.model().is_unused_checkpoint_weight
                    )
                elif kind == "namesake":
                    model = type(
                        "Qwen3_5MTP",
                        (),
                        {"load_weights": lambda self, weights: list(weights)},
                    )()
                config = SimpleNamespace(model=path, model_weights=None, revision=None)
                with (
                    patch.object(loader, "_prepare_weights", return_value=self.files),
                    patch.object(
                        runai,
                        "runai_safetensors_weights_iterator",
                        return_value=iter(()),
                    ) as stream,
                    patch.object(
                        runai,
                        "filter_safetensors_files_by_weight_name",
                        side_effect=AssertionError("unexpected selection"),
                    ),
                ):
                    loader.load_weights(model, config)
                stream.assert_called_once_with(self.files, False, distributed)

    def test_default_loader_selects_before_format_dispatch_and_prefix(self):
        for format_name, threaded, iterator in [
            ("auto", False, "safetensors_weights_iterator"),
            ("auto", True, "multi_thread_safetensors_weights_iterator"),
            ("instanttensor", False, "instanttensor_weights_iterator"),
            ("fastsafetensors", False, "fastsafetensors_weights_iterator"),
        ]:
            with self.subTest(format=format_name, threaded=threaded):
                loader = default.DefaultModelLoader.__new__(default.DefaultModelLoader)
                loader.load_config = SimpleNamespace(
                    load_format=format_name,
                    use_tqdm_on_load=False,
                    model_loader_extra_config={"enable_multithread_load": threaded},
                    safetensors_load_strategy="lazy",
                    safetensors_prefetch_num_threads=1,
                    safetensors_prefetch_block_size=1024,
                )
                loader._encoder_only_lm_prefixes = None
                loader.local_expert_ids = None
                source = default.DefaultModelLoader.Source(
                    str(self.root),
                    None,
                    prefix="prefixed.",
                    is_unused_weight=self.model().is_unused_checkpoint_weight,
                )
                with (
                    patch.object(
                        loader,
                        "_prepare_weights",
                        return_value=(str(self.root), self.files, True, "index.json"),
                    ),
                    patch.object(default, iterator, wraps=self.read) as stream,
                ):
                    result = dict(loader._get_weights_iterator(source))
                self.assertEqual(stream.call_args.args[0], self.files[1:])
                self.assertTrue(all(name.startswith("prefixed.") for name in result))
                self.assertNotIn("prefixed.model.visual.blocks.0.weight", result)

    def test_primary_hook_does_not_leak_into_secondary_sources(self):
        loader = default.DefaultModelLoader.__new__(default.DefaultModelLoader)
        model = self.model()
        model.secondary_weights = [
            default.DefaultModelLoader.Source("secondary", None, prefix="extra.")
        ]
        config = SimpleNamespace(model=str(self.root), revision=None)
        with patch.object(
            loader, "_get_weights_iterator", return_value=iter(())
        ) as iterator:
            list(loader.get_all_weights(config, model))
        primary, secondary = (call.args[0] for call in iterator.call_args_list)
        self.assertEqual(primary.is_unused_weight, model.is_unused_checkpoint_weight)
        self.assertIsNone(secondary.is_unused_weight)

    def test_default_loader_keeps_unfiltered_and_non_safetensors_sources(self):
        for use_safetensors, has_rule in ((True, False), (False, True)):
            with self.subTest(safetensors=use_safetensors, model_rule=has_rule):
                loader = default.DefaultModelLoader.__new__(default.DefaultModelLoader)
                loader.load_config = SimpleNamespace(
                    load_format="auto",
                    use_tqdm_on_load=False,
                    model_loader_extra_config={},
                    safetensors_load_strategy="lazy",
                    safetensors_prefetch_num_threads=1,
                    safetensors_prefetch_block_size=1024,
                    pt_load_map_location="cpu",
                )
                loader._encoder_only_lm_prefixes = None
                loader.local_expert_ids = None
                source = default.DefaultModelLoader.Source(
                    str(self.root),
                    None,
                    is_unused_weight=self.model().is_unused_checkpoint_weight
                    if has_rule
                    else None,
                )
                iterator = (
                    "safetensors_weights_iterator"
                    if use_safetensors
                    else "pt_weights_iterator"
                )
                with (
                    patch.object(
                        loader,
                        "_prepare_weights",
                        return_value=(
                            str(self.root),
                            self.files,
                            use_safetensors,
                            "index.json",
                        ),
                    ),
                    patch.object(default, iterator, return_value=iter(())) as stream,
                    patch.object(
                        default,
                        "filter_safetensors_files_by_weight_name",
                        side_effect=AssertionError("unexpected selection"),
                    ),
                ):
                    list(loader._get_weights_iterator(source))
                stream.assert_called_once()
                self.assertEqual(stream.call_args.args[0], self.files)


class SimpleNamespaceContext:
    """Expose only metadata to catch accidental payload reads during selection."""

    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        self.handle.__enter__()
        return SimpleNamespace(keys=self.handle.keys)

    def __exit__(self, *args):
        return self.handle.__exit__(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
