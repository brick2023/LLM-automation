"""
智慧的結晶
執行 conda 環境 data_collect
"""
from generate_QA.generate_QA import generate_QA
from modelTool.mediaKit import dir_to_text_and_srt_files
from modelTool.summarize import dir_long_text_to_summary_files
from modelTool.yt import yt_playlist_url_to_mp4
import os


def main():

    # 0. 設定路徑
    # course_name = "啾啾鞋"
    # playlist_url = "https://youtube.com/playlist?list=PLkHoI1J0zR8fBrYhl3HfcPfLz8CV_nhSb&si=SyrBRcbs40lpo9Ot"
    # course_name = "均一教育平台國中生物情境教學"
    # playlist_url = "https://youtube.com/playlist?list=PLp2Y5q36tB-OW1RcE2uLJIMUWL5xlCd2J&si=pOv5g9YR5eFctm1D"
    # course_name = "均一教育平台國中生物"
    # playlist_url = "https://youtube.com/playlist?list=PLp2Y5q36tB-PhGWqNOu_7unWfPg1t-sgq&si=oa3Tm7oRMVDvtGQ8"
    course_name = "國中公民"
    playlist_url = "https://youtube.com/playlist?list=PLm778hWdXOZmH0PF2PIE-M7HfzjZZokdR&si=WuG3_n8Jrj-Jk_TX"
    plain_text_dir_path = f"/home/brick2/platform2024/src/{course_name}/plain_text"
    srt_dir_path = f"/home/brick2/platform2024/src/{course_name}/srt"
    summary_dir_path = f"/home/brick2/platform2024/src/{course_name}/summay"
    video_dir_path = f"/home/brick2/platform2024/src/{course_name}/video"
    output_dataset_json_path = f"/home/brick2/platform2024/src/{course_name}/dataset.json"
    base_model = "yentinglin/Taiwan-LLM-7B-v2.0-base"
    llm_adapter_path = f"/home/brick2/platform2024/src/{course_name}/llm_adapter"
    finetune_script_path = f"/home/brick2/platform2024/LLM-automation/alpaca-lora/finetune.py"
    
    # 1. 下載影片
    print("下載影片")
    yt_playlist_url_to_mp4(playlist_url, video_dir_path)

    # 2. 生成文字檔和 srt 檔案
    print("生成文字檔和 srt 檔案")
    dir_to_text_and_srt_files(video_dir_path, plain_text_dir_path, srt_dir_path, model_size="large")

    # 3. 生成摘要檔案 (長文本摘要演算法)
    print("生成摘要檔案")
    dir_long_text_to_summary_files(plain_text_dir_path, summary_dir_path)

    # 4. 生成 QA 檔案
    run_times = 3
    print(f"生成 QA 檔案，{run_times}次迭代")
    import json
    # 嘗試讀取 output_dataset_json_path 到 json_obj，如果失敗，則初始化為空 list
    json_obj = []
    try:
        json_obj = json.load(open(output_dataset_json_path, "r"))
    except:
        json_obj = []
    for _ in range(run_times):
        generate_QA(plain_text_dir_path, output_dataset_json_path, summary_path=summary_dir_path)
        json_obj = json.load(open(output_dataset_json_path, "r")) + json_obj
        print("目前資料筆數:", len(json_obj))
        with open(output_dataset_json_path, "w") as f:
            json.dump(json_obj, f)

    # 5. 訓練模型 (如果資料量太少，可能會失敗，可以考慮用別的資料集) 參考：https://github.com/tloen/alpaca-lora/issues/470
    # TODO: 檢驗模型是否訓練成功
    print("訓練模型")
    llm_adapter_path = f"/home/brick2/platform2024/src/{course_name}/llm_adapter"
    prompt_template_name = "alpaca_tw"
    os.system(f"""cd alpaca-lora; 
              
              python {finetune_script_path}\
               --base_model '{base_model}'\
               --data_path '{output_dataset_json_path}'\
                --output_dir '{llm_adapter_path}'\
                --batch_size 32 \
                --micro_batch_size 16 \
                --num_epochs 20 \
                --learning_rate 1e-4 \
                --cutoff_len 512 \
                --val_set_size 10 \
                --lora_r 8 \
                --lora_alpha 16 \
                --lora_dropout 0.05 \
                --lora_target_modules '[q_proj,v_proj]' \
                --train_on_inputs \
                --group_by_length\
                --prompt_template_name='{prompt_template_name}'\
                ; cd ..
              """)

    # 6. 將 srt, summary, model, QA json, video 的路徑全部都加入資料庫
    print("將 srt, summary, model, QA json, video 的路徑全部都加入資料庫")
    os.system(f"""
              cd ..;
              python add_to_database.py --course_name {course_name} --playlist_url {playlist_url};
              cd LLM-automation
""")

if __name__ == "__main__":
    main()