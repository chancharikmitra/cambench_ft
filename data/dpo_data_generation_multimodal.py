#!/usr/bin/env python3
import os
import json
import torch
import torch.multiprocessing as mp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from PIL import Image

def worker(
    gpu_id: int,
    model_name: str,
    data: list,
    output_file: str,
    base_dir: str,
    num_gpus: int
):
    # Bind this process to a single GPU
    torch.cuda.set_device(gpu_id)

    # Load processor and model onto cuda:gpu_id
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": f"cuda:{gpu_id}"}
    )
    model.eval()

    # Determine this worker's slice of the data
    chunk_size = len(data) // num_gpus
    start = gpu_id * chunk_size
    end = len(data) if gpu_id == num_gpus - 1 else (start + chunk_size)
    my_chunk = data[start:end]

    results = []
    for item in tqdm(my_chunk, desc=f"[GPU {gpu_id}]"):
        try:
            user_msg = item["messages"][0]["content"]
            ref_out  = item["messages"][1]["content"]

            # Resolve media (video or image) under base_dir
            media_entries = []
            if videos := item.get("videos"):
                raw_vid = videos[0]
                vid_name = os.path.basename(raw_vid)
                full_vid = os.path.join(base_dir, vid_name)
                media_entries.append({"type": "video", "video": full_vid})
            elif images := item.get("images"):
                img_name = os.path.basename(images[0])
                full_img = os.path.join(base_dir, img_name)
                img = Image.open(full_img).convert("RGB")
                media_entries.append({"type": "image", "image": img})
            else:
                print(f"[GPU {gpu_id}] no media for item, skipping")
                continue

            # Build multimodal chat message
            messages = [{
                "role": "user",
                "content": media_entries + [{"type": "text", "text": user_msg}]
            }]

            # Prepare inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(f"cuda:{gpu_id}")

            # Generate response
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True
            )

            # Trim prompt tokens and decode
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
            ]
            gen_text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Assemble DPO example
            results.append({
                "conversations": [
                    {"from": "human", "value": user_msg}
                ],
                "chosen":   {"from": "gpt", "value": ref_out},
                "rejected": {"from": "gpt", "value": gen_text},
                "videos":   item.get("videos", []),
                "images":   item.get("images", [])
            })

        except Exception as e:
            print(f"[GPU {gpu_id}] error on item {item}: {e}")
            continue

    # Cleanup and write partial output
    del model, processor
    torch.cuda.empty_cache()

    part_file = output_file.replace(".json", f".gpu{gpu_id}.json")
    with open(part_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[GPU {gpu_id}] wrote {len(results)} examples to {part_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Configuration
    input_file  = "cam_motion_TRINITY/captionset.json"
    base_dir    = "/data3/zhiqiul/video_annotation/videos"
    output_file = "cam_motion_TRINITY/dpo_multimodal_ft_cambench.json"
    model_name  = "chancharikm/qwen2.5-vl-7b-cam-motion-preview"
    max_samples = 5000

    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)[:max_samples]
    # Optionally: data = data[:max_samples]

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs, spawning workers…")

    # Spawn one process per GPU
    mp.spawn(
        worker,
        args=(model_name, data, output_file, base_dir, num_gpus),
        nprocs=num_gpus,
        join=True
    )

    # Merge partial outputs
    all_results = []
    for gpu_id in range(num_gpus):
        part_file = output_file.replace(".json", f".gpu{gpu_id}.json")
        with open(part_file, "r", encoding="utf-8") as f:
            all_results.extend(json.load(f))
        os.remove(part_file)

    # Write final combined JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Merged total {len(all_results)} examples → {output_file}")
