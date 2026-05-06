import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


datasets = [
    'narrativeqa',
    'qasper',
    'multifieldqa_en',
    'hotpotqa',
    '2wikimqa',
    'musique',
    'comprehension_and_reasoning',
    'computation',
    'multiple_information_retrieval',
    'timeline_reorder'
]

# Multilingual LongBench (built via data/build_multilingual_longbench.py).
MLONGBENCH_DATASETS = [
    'xquad_en', 'xquad_de', 'xquad_zh', 'xquad_es', 'xquad_ar',
    'xquad_hi', 'xquad_vi', 'xquad_ru', 'xquad_th', 'xquad_tr',
    'mgsm_en', 'mgsm_de', 'mgsm_sw', 'mgsm_zh', 'mgsm_es',
    'mgsm_fr', 'mgsm_ja', 'mgsm_ru', 'mgsm_te', 'mgsm_th', 'mgsm_bn',
]


dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
    'comprehension_and_reasoning': 64,
    'multiple_information_retrieval': 64,
    'timeline_reorder': 32,
    'computation': 32,
}

# XQuAD answers are short spans; MGSM answers are short numerical strings.
for _ds in ('xquad_en', 'xquad_de', 'xquad_zh', 'xquad_es', 'xquad_ar',
            'xquad_hi', 'xquad_vi', 'xquad_ru', 'xquad_th', 'xquad_tr'):
    dataset2maxlen[_ds] = 48
for _ds in ('mgsm_en', 'mgsm_de', 'mgsm_sw', 'mgsm_zh', 'mgsm_es',
            'mgsm_fr', 'mgsm_ja', 'mgsm_ru', 'mgsm_te', 'mgsm_th', 'mgsm_bn'):
    dataset2maxlen[_ds] = 256

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    'comprehension_and_reasoning': 'Please answer the question based on the long texts below. \n{context}\nQuestion: {input}\nAnswer:',
    'computation': 'Please answer the question based on the long texts below. \n{context}\nQuestion: {input}\nAnswer:',
    'multiple_information_retrieval': 'Please answer the question based on the long texts below. \n{context}\nQuestion: {input}\nAnswer:',
    'timeline_reorder': 'Please answer the question based on the long texts below. \n{context}\nQuestion: {input}\nAnswer:',

    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

# Multilingual prompts. XQuAD-style: short extractive answer in the same
# language as the question. MGSM-style: solve the *target* problem only.
_XQUAD_PROMPTS = {
    'en': "Read the passages below and answer the question with a short span copied from the passages. Do not output any other words.\n\nPassages:\n{context}\n\nQuestion: {input}\nAnswer:",
    'de': "Lies die folgenden Absätze und beantworte die Frage mit einer kurzen Phrase, die wörtlich aus dem Text stammt. Gib keine zusätzlichen Wörter aus.\n\nAbsätze:\n{context}\n\nFrage: {input}\nAntwort:",
    'zh': "请阅读下面的段落，并用段落中出现的简短文字回答问题，不要输出任何其他内容。\n\n段落：\n{context}\n\n问题：{input}\n答案：",
    'es': "Lee los siguientes párrafos y responde la pregunta con una frase corta tomada del texto. No incluyas ninguna otra palabra.\n\nPárrafos:\n{context}\n\nPregunta: {input}\nRespuesta:",
    'ar': "اقرأ الفقرات التالية وأجب عن السؤال بعبارة قصيرة منقولة من النص. لا تكتب أي شيء آخر.\n\nالفقرات:\n{context}\n\nالسؤال: {input}\nالإجابة:",
    'hi': "नीचे दिए गए अनुच्छेदों को पढ़ें और प्रश्न का उत्तर पाठ से लिए गए संक्षिप्त वाक्यांश में दें। कोई अन्य शब्द न लिखें।\n\nअनुच्छेद:\n{context}\n\nप्रश्न: {input}\nउत्तर:",
    'vi': "Đọc các đoạn văn dưới đây và trả lời câu hỏi bằng một cụm từ ngắn lấy từ văn bản. Không xuất ra bất kỳ từ nào khác.\n\nVăn bản:\n{context}\n\nCâu hỏi: {input}\nTrả lời:",
    'ru': "Прочитайте отрывки ниже и ответьте на вопрос короткой фразой, взятой из текста. Не выводите ничего другого.\n\nОтрывки:\n{context}\n\nВопрос: {input}\nОтвет:",
    'th': "อ่านข้อความด้านล่างและตอบคำถามด้วยข้อความสั้นๆ ที่คัดมาจากเนื้อหา ห้ามพิมพ์อย่างอื่น\n\nข้อความ:\n{context}\n\nคำถาม: {input}\nคำตอบ:",
    'tr': "Aşağıdaki paragrafları oku ve soruyu metinden alınmış kısa bir ifade ile cevapla. Başka hiçbir şey yazma.\n\nParagraflar:\n{context}\n\nSoru: {input}\nCevap:",
}
_MGSM_PROMPTS = {
    'en': "You are given many math problems. Solve ONLY the target problem at the end. Output just the final numeric answer.\n\n{context}\n\nFinal numeric answer:",
    'de': "Dir werden viele Matheaufgaben gegeben. Löse NUR die Zielaufgabe am Ende. Gib nur die endgültige Zahl als Antwort aus.\n\n{context}\n\nEndgültige Zahl:",
    'zh': "下面给出了许多数学题。只回答最后标记的“目标题目”。只输出最终的数字答案。\n\n{context}\n\n最终数字答案：",
    'sw': "Umepewa matatizo mengi ya hesabu. Tatua TU swali lengo lililo mwishoni. Toa jibu la nambari pekee.\n\n{context}\n\nJibu la nambari:",
    'es': "Se te dan muchos problemas matemáticos. Resuelve SOLO el problema objetivo al final. Devuelve únicamente la respuesta numérica final.\n\n{context}\n\nRespuesta numérica final:",
    'fr': "Plusieurs problèmes de mathématiques te sont donnés. Résous UNIQUEMENT le problème cible à la fin. Donne uniquement la réponse numérique finale.\n\n{context}\n\nRéponse numérique finale :",
    'ja': "多くの算数の問題が与えられています。最後の「目標問題」だけを解いてください。最終的な数値だけを出力してください。\n\n{context}\n\n最終的な数値:",
    'ru': "Тебе даны много математических задач. Реши ТОЛЬКО целевую задачу в конце. Выведи только итоговое число.\n\n{context}\n\nИтоговое число:",
    'te': "మీకు చాలా గణిత సమస్యలు ఇవ్వబడ్డాయి. చివర్లో ఉన్న లక్ష్య సమస్యను మాత్రమే పరిష్కరించండి. చివరి సంఖ్యను మాత్రమే ఇవ్వండి.\n\n{context}\n\nచివరి సంఖ్య:",
    'th': "คุณได้รับโจทย์เลขจำนวนมาก จงแก้เฉพาะโจทย์เป้าหมายที่อยู่ตอนท้ายเท่านั้น ตอบเฉพาะตัวเลขสุดท้าย\n\n{context}\n\nตัวเลขคำตอบ:",
    'bn': "আপনাকে অনেক গণিত সমস্যা দেওয়া হয়েছে। শুধু শেষের লক্ষ্য সমস্যাটি সমাধান করুন। শুধু চূড়ান্ত সংখ্যা উত্তর হিসেবে দিন।\n\n{context}\n\nচূড়ান্ত সংখ্যা:",
}

for _ds in MLONGBENCH_DATASETS:
    if _ds.startswith('xquad_'):
        _lang = _ds.split('_', 1)[1]
        model2prompt[_ds] = _XQUAD_PROMPTS.get(_lang, _XQUAD_PROMPTS['en'])
    elif _ds.startswith('mgsm_'):
        _lang = _ds.split('_', 1)[1]
        model2prompt[_ds] = _MGSM_PROMPTS.get(_lang, _MGSM_PROMPTS['en'])


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500,
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

# def build_prompt(prompt, dataset):
    
#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):
    

    print("Loading data...")
    
    test_data = []
    
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
    input_max_len = 0
    
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
            

    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            
            length = example["length"]
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
                
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    
    for example in test_data:
        
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    
    model_name = model_path.split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")
     
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]
        
        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

        model.model.config.window_size = 8
        model.model.config.base_capacity = args.max_capacity_prompts
        model.model.config.head_choice = args.head_choice
        model.model.config.fuse_heads = args.fuse_heads
        model.model.config.beta = args.beta
        model.model.config.temp = args.temp
        
        model.model.config.kernel_size = 7
        model.model.config.skip = 0
        model.model.config.normalize = True
        model.model.config.pooling = "maxpool"
        model.model.config.floor = 0.2


        context_length = batch_input_ids.shape[-1]

        output = model.generate(
            **tokenized_prompts,
            output_attentions = args.output_attentions,
            max_new_tokens=output_max_len,
            num_beams=1,
            
            do_sample=False,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id]
        )


        batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)        
        batch_generations = batch_outputs

        torch.cuda.empty_cache()
        for j in range(args.eval_batch_size):
            
            example = {}
            
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["context"] = batch_contexts[j]
            example["answers"] = batch_answerss[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]
            
            example["dataset"] = batch_datasets[j]
            example["language"] = batch_languages[j]
            example["all_classes"] = batch_all_classess[j]
            example["_id"] = batch__ids[j]


            fout.write(json.dumps(example) + "\n")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")

    parser.add_argument(
        "--head_choice",
        type=str,
        default='random',
        choices=[
            'random', 'copy', 'reason',
            # single RTH (Retrieval-Transition Head) files
            'trans_ende', 'trans_ensw', 'trans_enzh', 'trans_zhen',
            # fusion modes (see headkv/snapkv_utils.py:FUSE_RECIPES)
            'fuse', 'fuse_rth', 'fuse_rth_zh', 'fuse_rth_all', 'rth_only',
        ],
    )
    parser.add_argument(
        "--fuse_heads",
        type=str,
        default=None,
        help=(
            "Only used when --head_choice=fuse. Comma-separated list of head-score "
            "keys to average, e.g. 'llama_copy,trans_ende,trans_ensw'. Keys are "
            "defined in headkv/snapkv_utils.py:HEAD_SCORE_FILES."
        ),
    )
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--temp', type=float, default=1.0)

    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of dataset names to override the default "
            "LongBench list. Use this to run the multilingual LongBench, e.g. "
            "'xquad_en,xquad_de,xquad_zh,mgsm_de,mgsm_sw'."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/LongBench",
        help="Directory containing <dataset>.jsonl files.",
    )
    
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.model_path == 'mistralai/Mistral-7B-Instruct-v0.2':
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left",
            revision='dca6e4b60aca009ed25ffa70c9bb65e46960a573'
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left"
        )

    if args.method.lower() != 'fullkv':
        from headkv.monkeypatch import replace_llama, replace_mistral 
        replace_llama(args.method)
        replace_mistral(args.method)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
    

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

        
    model.eval()
    
    save_dir = args.save_dir
    
        
    max_capacity_prompts = args.max_capacity_prompts
    



        

    if args.datasets:
        run_datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    else:
        run_datasets = datasets

    for idx, dataset in enumerate(run_datasets):

        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(run_datasets)}")
        print(f'base capacity: {args.max_capacity_prompts}\thead_choice:{args.head_choice}\tbeta:{args.beta}\ttemp:{args.temp}')

        args.dataset = dataset
        args.data_file = f"{args.data_dir}/{args.dataset}.jsonl"

        main(args)