import asyncio
from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
import numpy as np

async def test_inference_engine(inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str, n_layers: int):
  # breakpoint()
  prompt = "In a single word only, what is the last name of the current president of the USA?"
  resp_full, inference_state_full, _ = await inference_engine_1.infer_prompt("A", shard=Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers), prompt=prompt)

  print(f"{resp_full=}")

if __name__ == "__main__":
  asyncio.run(test_inference_engine(
    TinygradDynamicShardInferenceEngine(HFShardDownloader()),
    TinygradDynamicShardInferenceEngine(HFShardDownloader()),
    "meta-llama/Llama-3.2-1B-Instruct",
    16
  ))


