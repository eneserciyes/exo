from exo.inference.shard import Shard
from exo.inference.tinygrad.inference import build_transformer
from exo.download.hf.hf_shard_download import HFShardDownloader

async def main():
    shard = Shard(model_id="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", start_layer=0, end_layer=31, n_layers=32)
    model_path = await HFShardDownloader().ensure_shard(shard)

    llama_full = build_transformer(model_path, shard, "8B")

    print(llama_full)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())