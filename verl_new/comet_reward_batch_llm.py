# my_comet_reward.py  
from comet import download_model, load_from_checkpoint  

import re
import logging
from concurrent.futures import ThreadPoolExecutor
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)

from tqdm import tqdm
import torch
from openai import OpenAI
  
# 全局变量缓存模型  
_comet_model = None  


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
# models.list()返回一个模型列表，每个模型都有一个id属性
model_name = client.models.list().data[0].id

prompt = """You are an impartial evaluator of translation THINKING QUALITY.

Evaluate ONLY the thinking process for a general machine translation task.
Do NOT evaluate any final translation. Do NOT produce a translation.

Direction: {src_lang} → {tgt_lang}

Source sentence ({src_lang}):
{src}

Thinking text (verbatim, written by the model):
{think}

Scoring rubric (0–10 total; integers only).
Assign 0/1/2 points to each criterion below, then sum to get the final score (0–10).
1) coverage (0/1/2): Mentions key entities/numbers/terms from the source that must be preserved.
2) disambiguation (0/1/2): Identifies potential ambiguities/tricky parts and states a resolution or a check.
3) terminology_style (0/1/2): States terminology/proper-noun handling and desired register/style in {tgt_lang}.
4) reordering_plan (0/1/2): Provides a brief plan for segmentation/reordering (e.g., clause breaks, topicalization).
5) concision (0/1/2): Concise and non-redundant. Avoids restating the source verbatim; avoids generic templates.

Important judging notes:
- Judge ONLY the THINKING text’s usefulness for producing an accurate translation of the given source.
- If the thinking is generic (e.g., “I will translate this sentence…”), assign low points on concision and other relevant criteria.

OUTPUT REQUIREMENTS (CRITICAL):
Return ONLY a single JSON object (no extra text, no code fences) with EXACTLY these keys:
  score, coverage, disambiguation, terminology_style, reordering_plan, concision, explanation
Where:
- "score" is an integer 0..10 equal to (coverage + disambiguation + terminology_style + reordering_plan + concision).
- Each criterion field is an integer in {{0,1,2}}.
- "explanation" is a short one-paragraph rationale (in English), citing specific parts of the THINKING TEXT.
- If the THINKING TEXT is empty or unusable, output all zeros and a brief explanation.
example:
{{"score": 8, "coverage": 2, "disambiguation": 1, "terminology_style": 2, "reordering_plan": 1, "concision": 2, "explanation": "Your reason of evaluation"}}

Now produce the JSON for this sample.
"""
def call_api(src, think, src_lang="English", tgt_lang="Chinese"):
    prompt_text = prompt.format(src=src, think=think, src_lang=src_lang, tgt_lang=tgt_lang)
    # base64_image = encode_image(image)
    response = client.chat.completions.create(
        # model="模型",
        model = model_name, # 图文
        messages=[
            # {'role': 'system', 'content': system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                    ],
                }
        ],
    )
    return response.choices[0].message.content
  
def _load_comet_model():  
    global _comet_model  
    if _comet_model is None:  
        print("Loading COMET model...")
        
        # 在 Ray 多进程环境中重新初始化 CUDA 上下文
        try:
            # 尝试重新初始化 CUDA 环境
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            torch.cuda.init()
        except Exception as e:
            print(f"CUDA initialization warning: {e}")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            print(f"Available GPUs: {gpu_count}, Current device: {current_device}, GPU memory: {gpu_memory / 1e9:.1f}GB")
            
            # Load COMET model with device specification
            _comet_model = load_from_checkpoint("/mnt/data/users/liamding/data/models/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt")  
            # Move to specific GPU if available
            # Use a different GPU if multiple GPUs available to avoid conflict with vLLM
            target_device = f"cuda:{(current_device + 1) % gpu_count}" if gpu_count > 1 else f"cuda:{current_device}"
            _comet_model = _comet_model.to(target_device)
            print(f"Loaded COMET model on {target_device}")
        else:
            print(f"CUDA is not available: {torch.cuda.is_available()}")
            print("Loading COMET model on CPU (this will be slower)")
            _comet_model = load_from_checkpoint("/mnt/data/users/liamding/data/models/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt").to("cuda") 

    return _comet_model  

def compute_bleu(lg_pair, ref, pred):  
    import sacrebleu  
    import re  
      
    pred = pred if isinstance(pred, str) else ""  
    tgt_lang = lg_pair.split("-")[1]  
    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"  
      
    bleu = sacrebleu.sentence_bleu(pred, [ref], lowercase=True, tokenize=tokenize)
    return float(bleu.score)


def extract_solution(solution_str: str) -> str:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """

    answer_pattern = r'<translate>(.*?)</translate>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, None
    final_translate = matches[-1].group(1).strip()

    think_pattern = r'<think>(.*?)</think>'
    think_matches = list(re.finditer(think_pattern, solution_str, re.DOTALL))
    if not think_matches:
        print("[Error] No valid think tags found")
        return None, None
    final_think = think_matches[-1].group(1).strip()

    return final_translate, final_think


def extract_score_from_response(response_str: str) -> float:
    """Extracts score from the API response containing JSON with score field.
    
    Args:
        response_str: Raw response string from the thinking evaluation API
        
    Returns:
        Float score extracted from JSON, or 0.0 if not found
    """
    import json
    
    try:
        # 尝试直接解析JSON
        if response_str.strip().startswith('{') and response_str.strip().endswith('}'):
            data = json.loads(response_str.strip())
            score = data.get('score', 0)
            # 将分数标准化到0-1范围（假设原始分数是0-10）
            normalized_score = float(score) / 10.0
            return normalized_score
    except json.JSONDecodeError:
        pass
    
    # 备用方案：查找JSON代码块
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.finditer(json_pattern, response_str, re.DOTALL)
    
    for match in matches:
        try:
            json_str = match.group(1)
            data = json.loads(json_str)
            score = data.get('score', 0)
            normalized_score = float(score) / 10.0
            return normalized_score
        except json.JSONDecodeError:
            continue
    
    # 再次备用方案：查找裸JSON对象
    json_pattern2 = r'\{[^}]*"score"[^}]*\}'
    matches2 = re.finditer(json_pattern2, response_str, re.DOTALL)
    
    for match in matches2:
        try:
            json_str = match.group(0)
            data = json.loads(json_str)
            score = data.get('score', 0)
            normalized_score = float(score) / 10.0
            return normalized_score
        except json.JSONDecodeError:
            continue
    
    print(f"[Warning] No valid score found in response: {response_str}...")
    return 0.0


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<translate>', 1),
        'answer_end': ('</translate>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        # print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    # else:
    #     print("  Tag sequence validation passed")

    return validation_passed


def compute_score_single(data_source, solution_str, ground_truth, extra_info=None):  
    """
    Single-item version of compute_score function for backward compatibility.
    Used by NaiveRewardManager.
    """
    lg_pair = extra_info.get("lg", "en-zh") if extra_info else "en-zh"  
    src_text = extra_info.get("source", ground_truth) if extra_info else ground_truth  
    
    format_score = validate_response_structure(solution_str)
    
    if not format_score:  
        print("invalid format")
        return -3.0  # 格式错误惩罚，与batch版本保持一致  
    
    answer_text, think_text = extract_solution(solution_str)
    if answer_text is  None and think_text is None:
        print("format score is 1.0 but no <translate> tag found in completion: ", solution_str)
        return -3.0

    bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)  
    
    model = _load_comet_model()  
    comet_data = [{"src": src_text, "mt": answer_text}]  
    comet_scores = model.predict(comet_data, batch_size=8, gpus=1, progress_bar=False).scores  
    comet_score = float(comet_scores[0])  # 直接用原始分数
    final_score = format_score + (bleu_score / 100.0) + comet_score  # BLEU缩放，COMET不缩放
    print("final score: ", final_score)
    
    return final_score


def _forward_micro_batch(model, micro_batch):
    """
    Process a micro batch of triplets using COMET model.
    Migrated from MT-R1-Zero DataParallelCOMET._forward_micro_batch
    """
    batch_size = len(micro_batch)
    print(f"comet_reward.py forward micro_batch: {batch_size}")
    comet_output = model.predict(micro_batch, batch_size=batch_size, gpus=1, progress_bar=False)
    # 直接返回原始分数，不再*100
    scores = [float(score) for score in comet_output.scores]
    return scores

def compute_score(*args, **kwargs):
    """
    Adaptive compute_score function that supports both single and batch processing.
    
    For single processing (NaiveRewardManager):
        compute_score(data_source, solution_str, ground_truth, extra_info=None)
        
    For batch processing (BatchRewardManager):
        compute_score(data_sources=[], solution_strs=[], ground_truths=[], extra_infos=None, ...)
    """
    # Check if this is a batch call (BatchRewardManager style)
    if 'data_sources' in kwargs or 'solution_strs' in kwargs or 'ground_truths' in kwargs:
        # Batch processing call
        data_sources = kwargs.get('data_sources', [])
        solution_strs = kwargs.get('solution_strs', [])
        ground_truths = kwargs.get('ground_truths', [])
        extra_infos = kwargs.get('extra_infos', None)
        micro_batch_size = kwargs.get('micro_batch_size', 8)
        
        print(f"Using BATCH processing for {len(solution_strs)} items")
        return compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, micro_batch_size)
    
    # Check if this is positional arguments (single processing)
    elif len(args) >= 3:
        # Single processing call
        print("Using SINGLE processing for 1 item")
        data_source = args[0]
        solution_str = args[1] 
        ground_truth = args[2]
        extra_info = args[3] if len(args) > 3 else kwargs.get('extra_info', None)
        
        return compute_score_single(data_source, solution_str, ground_truth, extra_info)
    
    # Check if this is single item with keyword arguments (DAPO style)
    elif 'data_source' in kwargs and 'solution_str' in kwargs and 'ground_truth' in kwargs:
        # Single item with keyword arguments
        print("Using SINGLE processing for 1 item (keyword args)")
        data_source = kwargs['data_source']
        solution_str = kwargs['solution_str']
        ground_truth = kwargs['ground_truth']
        extra_info = kwargs.get('extra_info', None)
        
        return compute_score_single(data_source, solution_str, ground_truth, extra_info)

    
    else:
        raise ValueError(f"Invalid arguments for compute_score: args={args}, kwargs={kwargs}")

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None, micro_batch_size=8):
    """
    Batch version of compute_score function.
    Migrated and optimized from MT-R1-Zero DataParallelCOMET.compute_comet_rm
    
    Args:
        data_sources: List of data sources
        solution_strs: List of solution strings
        ground_truths: List of ground truth strings
        extra_infos: List of extra info dicts (optional)
        micro_batch_size: Size of micro batches for COMET processing
        
    Returns:
        List of final scores
    """
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)
    
    triplet_list = []
    final_scores = []
    
    print(f"Processing batch of {len(solution_strs)} items...")
    print("data_sources", len(data_sources), "solution_strs", len(solution_strs),
          "ground_truths", len(ground_truths), "extra_infos", len(extra_infos))

    model = _load_comet_model()
    
    invalid_items=[]
    for i in tqdm(range(len(solution_strs)), desc="checking format and building triplets"):
        data_source = data_sources[i]
        solution_str = solution_strs[i]
        ground_truth = ground_truths[i]
        extra_info = extra_infos[i]
        
        lg_pair = extra_info.get("lg", "en-zh") if extra_info else "en-zh"
        src_text = extra_info.get("source", ground_truth) if extra_info else ground_truth
        
        format_score = validate_response_structure(solution_str)
        if not format_score:
            invalid_items.append(i)
            # final_scores.append(-3.0)
            final_scores.append(0)
            continue
        
        answer_text = extract_solution(solution_str)
        if answer_text is None:
            invalid_items.append(i)
            # final_scores.append(-3.0)
            final_scores.append(0)
            continue
        answer_text, think_text = extract_solution(solution_str)
        if answer_text is  None and think_text is None:
            invalid_items.append(i)
            # final_scores.append(-3.0)
            final_scores.append(0)
            continue
        
        bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)
        # bleu_score = 0
        
        triplet_item = {"src": src_text, "mt": answer_text}
        think_src_item = {"src": src_text, "think": think_text,"lg_pair": lg_pair}
        triplet_list.append({
            "index": i,
            "triplet": triplet_item,
            "format_score": format_score,
            "bleu_score": bleu_score,
            "thinking": think_src_item
        })
    print(f"invalid items number {len(invalid_items)} / {len(solution_strs)}")
    
    if triplet_list:
        comet_triplets = [item["triplet"] for item in triplet_list]
        print("Processing comet triplets", len(comet_triplets), comet_triplets[:2])

        # 同时处理COMET和thinking评估
        think_triplets = [item["thinking"] for item in triplet_list]
        print(f"Processing COMET scores for {len(comet_triplets)} items...")
        print(f"Processing thinking evaluation for {len(think_triplets)} items...")
        
        # 并行获取COMET分数和thinking评估结果
        # COMET评估
        scores = model.predict(comet_triplets, batch_size=32, gpus=1)
        comet_scores_flat = [float(score) for score in scores.scores]
        
        # Thinking评估（并行处理）
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for item in think_triplets:
                # 直接从item中获取语言信息
                lg_pair = item.get("lg_pair", "en-zh")
                src_lang, tgt_lang = lg_pair.split("-")
                src_lang_name = {"en": "English", "zh": "Chinese", "ja": "Japanese", "de": "German", "fr": "French"}.get(src_lang, src_lang)
                tgt_lang_name = {"en": "English", "zh": "Chinese", "ja": "Japanese", "de": "German", "fr": "French"}.get(tgt_lang, tgt_lang)
                
                future = executor.submit(call_api, item["src"], item["think"], src_lang_name, tgt_lang_name)
                futures.append(future)
            
            # 使用tqdm显示thinking评估进度
            thinking_results = []
            for future in tqdm(futures, desc="Thinking evaluation progress"):
                thinking_results.append(future.result())
        
        # 一次性计算所有分数
        for i, (item, thinking_result) in enumerate(zip(triplet_list, thinking_results)):
            original_index = item["index"]
            format_score = item["format_score"]
            bleu_score = item["bleu_score"]
            comet_score = comet_scores_flat[i]  # 不再/100.0
            thinking_score = extract_score_from_response(thinking_result)
            
            # 计算最终分数（一次完成所有组件）
            # final_score = format_score + (bleu_score / 100.0) + comet_score + thinking_score
            final_score =  (bleu_score / 100.0) + comet_score + thinking_score
            
            # 确保final_scores列表足够长
            while len(final_scores) <= original_index:
                final_scores.append(0.0)
            final_scores[original_index] = final_score
            
            print(f"Item {original_index}: final_score={final_score} (format={format_score}, bleu={bleu_score/100.0:.3f}, comet={comet_score:.3f}, thinking={thinking_score:.3f})")




    while len(final_scores) < len(solution_strs):
        # final_scores.append(-3.0)
        final_scores.append(0.0)
    print(f"Batch processing completed: {len(final_scores)} scores computed")
    return final_scores