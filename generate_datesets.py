import json
import argparse
import numpy as np
from transformers import PreTrainedTokenizerFast
from concurrent.futures import ProcessPoolExecutor

MODEL_PATH = "/nvme1n1/DeepSeek-V4-Pro-w4a8-mtp"
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
    parser.add_argument('--workers', type=int, default=10, help='并行worker数，大数据集建议4-8（默认: 1）')
    return parser.parse_args()

_global_tokenizer = None
_global_safe_ids = None

def _init_worker(model_path, safe_ids_list):
    global _global_tokenizer, _global_safe_ids
    _global_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    _global_safe_ids = np.array(safe_ids_list)

def gen_exact_tokens(target_tokens, tokenizer, safe_ids, rng):
    if target_tokens == 0:
        return ""
    buffer = max(50, int(target_tokens * 0.15))
    token_ids = rng.choice(safe_ids, target_tokens + buffer).tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    actual_ids = tokenizer.encode(text, add_special_tokens=False)

    for _ in range(10):
        actual_count = len(actual_ids)
        if actual_count == target_tokens:
            return text
        if actual_count > target_tokens:
            actual_ids = actual_ids[:target_tokens]
            text = tokenizer.decode(actual_ids, skip_special_tokens=True)
            actual_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            need = target_tokens - actual_count
            extra = need + max(50, int(need * 0.15))
            more_ids = rng.choice(safe_ids, extra).tolist()
            actual_ids = actual_ids + more_ids
            text = tokenizer.decode(actual_ids, skip_special_tokens=True)
            actual_ids = tokenizer.encode(text, add_special_tokens=False)

    return text

def _generate_request(args):
    idx, common_prefix, prefix_tokens, suffix_tokens, seed = args
    rng = np.random.default_rng(seed + idx + 1)
    tokenizer = _global_tokenizer
    safe_ids = _global_safe_ids

    private_prefix = gen_exact_tokens(prefix_tokens, tokenizer, safe_ids, rng)
    private_suffix = gen_exact_tokens(suffix_tokens, tokenizer, safe_ids, rng)

    common_entry = {"question": common_prefix, "answer": "test"}
    prefill_question = (common_prefix + " " + private_prefix) if common_prefix else private_prefix
    prefill_entry = {"question": prefill_question, "answer": "test"}
    full_question = (prefill_question + " " + private_suffix) if prefill_question else private_suffix
    full_entry = {"question": full_question, "answer": "test"}

    return (common_entry, prefill_entry, full_entry)

def save_jsonl(data: list, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

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
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)
    valid_ids = [i for i in range(vocab_size) if i not in special_ids]

    print("筛选roundtrip安全token...")
    safe_ids = []
    for tid in valid_ids:
        decoded = tokenizer.decode([tid], skip_special_tokens=True)
        if decoded and len(tokenizer.encode(decoded, add_special_tokens=False)) == 1:
            safe_ids.append(tid)
    print(f"  安全token: {len(safe_ids)} / {len(valid_ids)}")

    if len(safe_ids) == 0:
        print("警告: 无可用安全token，回退到valid_ids")
        safe_ids = valid_ids

    safe_ids_np = np.array(safe_ids)
    rng = np.random.default_rng(SEED)

    common_prefix = ""
    if COMMON_PREFIX_TOKENS > 0:
        common_prefix = gen_exact_tokens(COMMON_PREFIX_TOKENS, tokenizer, safe_ids_np, rng)
        print(f"已生成公共前缀: {COMMON_PREFIX_TOKENS} tokens")

    common_data = []
    prefill_data = []
    full_data = []

    if args.workers <= 1:
        for i in range(NUM_REQUESTS):
            request_rng = np.random.default_rng(SEED + i + 1)
            private_prefix = gen_exact_tokens(PRIVATE_PREFIX_TOKENS, tokenizer, safe_ids_np, request_rng)
            private_suffix = gen_exact_tokens(PRIVATE_SUFFIX_TOKENS, tokenizer, safe_ids_np, request_rng)

            common_data.append({"question": common_prefix, "answer": "test"})
            prefill_question = (common_prefix + " " + private_prefix) if common_prefix else private_prefix
            prefill_data.append({"question": prefill_question, "answer": "test"})
            full_question = (prefill_question + " " + private_suffix) if prefill_question else private_suffix
            full_data.append({"question": full_question, "answer": "test"})

            common_len = len(tokenizer.encode(common_prefix, add_special_tokens=False)) if common_prefix else 0
            prefill_len = len(tokenizer.encode(prefill_question, add_special_tokens=False))
            full_len = len(tokenizer.encode(full_question, add_special_tokens=False))
            print(f"prompt [{i+1}]: common={common_len} prefill={prefill_len} full={full_len}")


    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(MODEL_PATH, safe_ids)) as pool:
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
