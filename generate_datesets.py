import json
import argparse
import os
import pickle
import platform
import multiprocessing
import numpy as np
from transformers import PreTrainedTokenizerFast
from concurrent.futures import ProcessPoolExecutor

# MODEL_PATH = "/nvme1n1/DeepSeek-V4-Pro-w4a8-mtp"
MODEL_PATH = "/nvme1n1/models/LLM/GLM-5.1-w8a8"
NUM_REQUESTS = 50
PRIVATE_PREFIX_TOKENS = 3584
PRIVATE_SUFFIX_TOKENS = 3584

def parse_args():
    parser = argparse.ArgumentParser(description='生成KV Cache测试数据集')
    parser.add_argument('--seed', type=int, default=100, help='随机种子（默认: 100）')
    parser.add_argument('--num-requests', type=int, default=50, help='请求数量（默认: 50）')
    parser.add_argument('--private-prefix', type=int, default=500, help='私有前缀token数（默认: 500）')
    parser.add_argument('--private-suffix', type=int, default=500, help='私有后缀token数（默认: 500）')
    parser.add_argument('--common-prefix', type=int, default=0, help='公共前缀token数，所有请求共享（默认: 0）')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH, help='模型路径')
    parser.add_argument('--output-common', type=str, default='common.jsonl', help='公共前缀数据集文件名')
    parser.add_argument('--output-prefill', type=str, default='prefill.jsonl', help='预埋数据集文件名')
    parser.add_argument('--output-full', type=str, default='full.jsonl', help='完整数据集文件名')
    parser.add_argument('--workers', type=int, default=256, help='并行worker数（默认: 20）')
    return parser.parse_args()

_global_tokenizer = None
_global_sample_pool = None

def _init_worker(model_path, sample_pool_list):
    global _global_tokenizer, _global_sample_pool
    if _global_tokenizer is None:
        _global_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    if _global_sample_pool is None:
        _global_sample_pool = np.array(sample_pool_list)

def gen_exact_tokens(target_tokens, tokenizer, sample_pool, rng):
    if target_tokens == 0:
        return ""
    sampled_ids = rng.choice(sample_pool, target_tokens).tolist()
    text = tokenizer.decode(sampled_ids, skip_special_tokens=True)
    actual_ids = tokenizer.encode(text, add_special_tokens=False)

    for _ in range(10):
        actual_count = len(actual_ids)
        if actual_count == target_tokens:
            return text
        if actual_count > target_tokens:
            excess = actual_count - target_tokens
            remove_est = max(1, int(excess * len(sampled_ids) / actual_count))
            sampled_ids = sampled_ids[:len(sampled_ids) - remove_est]
            text = tokenizer.decode(sampled_ids, skip_special_tokens=True)
            actual_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            need = target_tokens - actual_count
            extra = need + max(20, int(need * 0.1))
            more_ids = rng.choice(sample_pool, extra).tolist()
            sampled_ids = sampled_ids + more_ids
            text = tokenizer.decode(sampled_ids, skip_special_tokens=True)
            actual_ids = tokenizer.encode(text, add_special_tokens=False)

    return text

def _generate_request(args):
    idx, common_prefix, prefix_tokens, suffix_tokens, seed = args
    rng = np.random.default_rng(seed + idx + 1)
    tokenizer = _global_tokenizer
    sample_pool = _global_sample_pool

    private_prefix = gen_exact_tokens(prefix_tokens, tokenizer, sample_pool, rng)
    private_suffix = gen_exact_tokens(suffix_tokens, tokenizer, sample_pool, rng)

    common_entry = {"question": common_prefix, "answer": "test"}
    prefill_question = (common_prefix + " " + private_prefix) if common_prefix else private_prefix
    prefill_entry = {"question": prefill_question, "answer": "test"}
    full_question = (prefill_question + " " + private_suffix) if prefill_question else private_suffix
    full_entry = {"question": full_question, "answer": "test"}

    return (common_entry, prefill_entry, full_entry)

def save_jsonl(data: list, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(item, ensure_ascii=False) for item in data) + "\n")

def main():
    args = parse_args()

    SEED = args.seed
    NUM_REQUESTS = args.num_requests
    PRIVATE_PREFIX_TOKENS = args.private_prefix
    PRIVATE_SUFFIX_TOKENS = args.private_suffix
    COMMON_PREFIX_TOKENS = args.common_prefix
    MODEL_PATH = args.model_path

    print(f"配置信息:")
    print(f"  随机种子: {SEED}")
    print(f"  请求数量: {NUM_REQUESTS}")
    print(f"  公共前缀tokens: {COMMON_PREFIX_TOKENS}")
    print(f"  私有前缀tokens: {PRIVATE_PREFIX_TOKENS}")
    print(f"  私有后缀tokens: {PRIVATE_SUFFIX_TOKENS}")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  并行workers: {args.workers}")

    np.random.seed(SEED)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)

    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "safe_ids_cache.pkl")
    safe_ids = None
    word_start_ids = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("model_path") == MODEL_PATH and cache.get("vocab_size") == tokenizer.vocab_size:
                safe_ids = cache["safe_ids"]
                word_start_ids = cache.get("word_start_ids")
                print(f"从缓存加载安全token: {len(safe_ids)} 个 (跳过筛选)")
                if word_start_ids:
                    print(f"  词首token: {len(word_start_ids)} 个")
                else:
                    print("  缓存缺少词首token，从safe_ids补算...")
                    token_strs = tokenizer.convert_ids_to_tokens(safe_ids)
                    word_start_ids = [tid for tid, ts in zip(safe_ids, token_strs) if ts and (ts.startswith('Ġ') or ts.startswith('▁'))]
                    print(f"  词首token: {len(word_start_ids)} 个")
                    if len(word_start_ids) < 100:
                        print("警告: 词首token太少，回退到safe_ids")
                        word_start_ids = safe_ids
                    with open(cache_path, "wb") as f:
                        pickle.dump({"model_path": MODEL_PATH, "vocab_size": tokenizer.vocab_size, "safe_ids": safe_ids, "word_start_ids": word_start_ids}, f)
        except Exception:
            safe_ids = None
            word_start_ids = None

    if safe_ids is None:
        vocab_size = tokenizer.vocab_size
        special_ids = set(tokenizer.all_special_ids)
        valid_ids = [i for i in range(vocab_size) if i not in special_ids]

        print("筛选roundtrip安全token（批量模式）...")
        decoded_batch = tokenizer.batch_decode([[tid] for tid in valid_ids], skip_special_tokens=True)
        non_empty_pairs = [(tid, s) for tid, s in zip(valid_ids, decoded_batch) if s]
        strings_to_check = [s for _, s in non_empty_pairs]
        encodings = tokenizer.backend_tokenizer.encode_batch(strings_to_check, add_special_tokens=False)
        safe_ids = [tid for (tid, _), enc in zip(non_empty_pairs, encodings) if len(enc.ids) == 1]
        print(f"  安全token: {len(safe_ids)} / {len(valid_ids)}")

        if len(safe_ids) == 0:
            print("警告: 无可用安全token，回退到valid_ids")
            safe_ids = valid_ids

        print("筛选词首token（Ġ/▁开头，边界不合并）...")
        token_strs = tokenizer.convert_ids_to_tokens(safe_ids)
        word_start_ids = [tid for tid, ts in zip(safe_ids, token_strs) if ts and (ts.startswith('Ġ') or ts.startswith('▁'))]
        print(f"  词首token: {len(word_start_ids)} / {len(safe_ids)}")

        if len(word_start_ids) < 100:
            print("警告: 词首token太少，回退到safe_ids")
            word_start_ids = safe_ids

        with open(cache_path, "wb") as f:
            pickle.dump({"model_path": MODEL_PATH, "vocab_size": vocab_size, "safe_ids": safe_ids, "word_start_ids": word_start_ids}, f)
        print(f"  已缓存到 {cache_path}")

    sample_pool = word_start_ids if word_start_ids else safe_ids
    sample_pool_np = np.array(sample_pool)
    print(f"采样池: {len(sample_pool)} 个token (词首={len(word_start_ids) if word_start_ids else 0}, 全部safe={len(safe_ids)})")
    rng = np.random.default_rng(SEED)

    common_prefix = ""
    if COMMON_PREFIX_TOKENS > 0:
        common_prefix = gen_exact_tokens(COMMON_PREFIX_TOKENS, tokenizer, sample_pool_np, rng)
        print(f"已生成公共前缀: {COMMON_PREFIX_TOKENS} tokens")

    common_data = []
    prefill_data = []
    full_data = []

    if args.workers <= 1:
        for i in range(NUM_REQUESTS):
            request_rng = np.random.default_rng(SEED + i + 1)
            private_prefix = gen_exact_tokens(PRIVATE_PREFIX_TOKENS, tokenizer, sample_pool_np, request_rng)
            private_suffix = gen_exact_tokens(PRIVATE_SUFFIX_TOKENS, tokenizer, sample_pool_np, request_rng)

            common_data.append({"question": common_prefix, "answer": "test"})
            prefill_question = (common_prefix + " " + private_prefix) if common_prefix else private_prefix
            prefill_data.append({"question": prefill_question, "answer": "test"})
            full_question = (prefill_question + " " + private_suffix) if prefill_question else private_suffix
            full_data.append({"question": full_question, "answer": "test"})

            prefix_len = len(tokenizer.encode(private_prefix, add_special_tokens=False))
            suffix_len = len(tokenizer.encode(private_suffix, add_special_tokens=False))
            full_len = len(tokenizer.encode(prefill_question + " " + private_suffix if prefill_question else private_suffix, add_special_tokens=False))
            print(f"  [{i+1}] prefix={prefix_len} suffix={suffix_len} full={full_len}")

    else:
        mp_ctx = multiprocessing.get_context('fork') if platform.system() == 'Linux' else None
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(MODEL_PATH, sample_pool), mp_context=mp_ctx) as pool:
            task_args = [(i, common_prefix, PRIVATE_PREFIX_TOKENS, PRIVATE_SUFFIX_TOKENS, SEED) for i in range(NUM_REQUESTS)]
            results = list(pool.map(_generate_request, task_args))

        common_data = [r[0] for r in results]
        prefill_data = [r[1] for r in results]
        full_data = [r[2] for r in results]
        print(f"  已完成 {NUM_REQUESTS}/{NUM_REQUESTS}")

    save_jsonl(common_data, args.output_common)
    save_jsonl(prefill_data, args.output_prefill)
    save_jsonl(full_data, args.output_full)
    total_prefix = COMMON_PREFIX_TOKENS + PRIVATE_PREFIX_TOKENS
    print(f"\n数据集生成完成！")
    print(f"  {args.output_common}: {NUM_REQUESTS}条 × {COMMON_PREFIX_TOKENS} tokens (公共前缀)")
    print(f"  {args.output_prefill}: {NUM_REQUESTS}条 × {total_prefix} tokens (公共前缀+私有前缀)")
    print(f"  {args.output_full}: {NUM_REQUESTS}条 × {total_prefix + PRIVATE_SUFFIX_TOKENS} tokens (公共前缀+私有前缀+私有后缀)")

if __name__ == "__main__":
    main()


# python generate_datasets.py --num-requests 2000 --common-prefix 0 --private-prefix 4096 --private-suffix 4096 --seed 140000
# sed -i 's/max_out_len=[0-9]\+/max_out_len=1/' ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py
# sed -i 's/request_rate=[0-9.]\+/request_rate=0/' ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py
# sed -i 's/batch_size=[0-9]\+/batch_size=256/' ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py
# # echo "存储预埋"
# # ais_bench --models vllm_api_general_stream --custom-dataset-path prefill.jsonl --mode perf  --num-warmups 0  --num-prompts 10
# python hit_rate.py --action before --pods "10.141.19.144:8001" "10.141.19.144:8002" "10.141.19.145:8001" "10.141.19.145:8002"
# echo "正式测试"
# sed -i 's/max_out_len=[0-9]\+/max_out_len=512/' ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py
# ais_bench --models vllm_api_general_stream --custom-dataset-path full.jsonl --mode perf --num-warmups 0  --num-prompts 10
# python hit_rate.py --action after --pods "10.141.19.144:8001" "10.141.19.144:8002" "10.141.19.145:8001" "10.141.19.145:8002"
