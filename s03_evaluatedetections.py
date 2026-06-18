import os
import json
from glob import glob
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from functools import lru_cache
from sentence_transformers import util

THRESHOLD_IOU = 0.3 
SEMANTIC_THRESHOLD = 0.5 
model = SentenceTransformer('all-MiniLM-L6-v2')

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0



# Wrap only the expensive part: computing similarity
@lru_cache(maxsize=50000)
def _cached_similarity(text1, text2, threshold):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return similarity >= threshold, similarity

@lru_cache(maxsize=50000)
def open_image(path):
    return Image.open(path)

def are_semantically_similar(text1, text2, threshold=0.5):
    if text1 is None or text2 is None:
        return False, 0.0
    return _cached_similarity(text1, text2, threshold)

def enhance_sets(tup):
    set1, set2 = tup
    SET1 = set()
    for s1 in set1:
        similar_item = s1
        for s2 in set2:
            if are_semantically_similar(s1,s2):
                similar_item = s2
                break
        SET1.add(similar_item)
    return (SET1,set2)

def get_info(element_path):
    elements = json.load(open(element_path))
    sample = [i for i in elements if i['element_type']=='Icon'][0]
    out_dict = {key:[] for key in sample.keys()}
    for element in elements:
        for key in out_dict.keys():
            info = element[key] if key in element else None
            out_dict[key].append(info)
    return pd.DataFrame(out_dict)

def is_element_match(rowA, rowB, iou_threshold=THRESHOLD_IOU, semantic_threshold=SEMANTIC_THRESHOLD, consider_overlap = True, normalized_box=False):
    """
    Returns True if:
      1. dictA['represents'] and dictB['represents'] are semantically similar
      2. IoU(dictA['bounding_box'], dictB['bounding_box']) >= iou_threshold
    """
    repA = rowA["represents"]
    repB = rowB["represents"]
    boxA = rowA["reconst_bounding_box"] if normalized_box else rowA['bounding_box']
    boxB = rowB["bounding_box"]

    if rowA['element_type'] != rowB['element_type']:
        return False
    if rowA['element_type'] == 'Icon':
        sem_ok, sem_score = are_semantically_similar(repA, repB, threshold=semantic_threshold)
        if not sem_ok:
            return False
    overlap = iou(boxA, boxB)
    if consider_overlap:
        if overlap < iou_threshold:
            return False
    return True

def get_pred_path(pipeline_name, fname):
    pred_path1 = os.path.join(
            data_dir.replace("01_synthetic_data", f"02_detected_picture_elements/{pipeline_name}"),
            fname.replace("gt", "detected")
    )
    pred_path2 = os.path.join(
        data_dir.replace("01_synthetic_data", f"02_detected_picture_elements/{pipeline_name}"),
        fname.replace("_gt", "")
    )
    pred_path = pred_path1 if os.path.exists(pred_path1) else pred_path2
    return pred_path

def get_matchings(pred_df, gt_df, consider_overlap, normalized_box=False):
    tp, fp, fn = [], [], []
    gt_matched = []
    for pred_ind, pred_row in pred_df.iterrows():
        match_found = False
        for gt_ind, gt_row in gt_df.iterrows():
            if is_element_match(pred_row, gt_row, consider_overlap=consider_overlap, normalized_box=normalized_box):
                merged = pd.concat([pred_row, gt_row], axis=0)
                merged = merged.to_frame().T  # make it a single-row DataFrame
                tp.append(merged)
                match_found = True
                gt_matched.append(gt_ind)
        if not match_found:
            fp.append(pred_row.to_frame().T)
    for gt_ind, gt_row in gt_df.iterrows():
        if gt_ind not in gt_matched:
            fn.append(gt_row.to_frame().T)
    
    tp_df = pd.concat(tp, ignore_index=True) if len(tp)>0 else pd.DataFrame()
    fp_df = pd.concat(fp, ignore_index=True) if len(fp)>0 else pd.DataFrame()
    fn_df = pd.concat(fn, ignore_index=True) if len(fn)>0 else pd.DataFrame()
    return tp_df, fp_df, fn_df

def get_metrics(tp_df,fp_df,fn_df):
    tp = len(tp_df)
    fp = len(fp_df)
    fn = len(fn_df)
    try:
        precision = (tp / (tp + fp))
    except:
        precision = -1
    try:
        recall = tp / (tp + fn)
    except: 
        recall = -1
    return tp, fp, fn, precision, recall

def get_summary(result_df):
    summary_dict = {'metric':[], 'value':[]}
    for key, df in result_df.items():
        prefix = "not " if not key else ""
        for metric in ['precision', 'recall']:
            summary_dict["metric"].append(f"{prefix}considering bbox mean. {metric}")
            summary_dict["value"].append(df[metric].mean())
        summary_dict["metric"].append(f"{prefix}considering bbox dataset precision") 
        try:
            summary_dict["value"].append(df['tp'].sum()/(df['tp'].sum()+df['fp'].sum()))
        except:
            summary_dict["value"].append(-1)
        summary_dict["metric"].append(f"{prefix}considering bbox dataset recall")
        try:
            summary_dict["value"].append(df['tp'].sum()/(df['tp'].sum()+df['fn'].sum()))
        except:
            summary_dict["value"].append(-1)
    summary_df = pd.DataFrame(summary_dict)
    return summary_df

def evaluate(data_dir, pipeline_name, normalized_box=False, target_element_type = 'Icon'):
    global SUPER_SUMMARY
    result_dict = {}
    result_df = {}
    tp_dfs = {True:[], False:[]}
    fp_dfs = {True:[], False:[]}
    fn_dfs = {True:[], False:[]}
    
    for fname in sorted(os.listdir(data_dir)):
        print(f"evaluating {fname}")
        if (not fname.endswith(".json")) or ("data_summary" in fname):
            continue

        base = fname.replace(".json", "")
        gt_path = os.path.join(data_dir, f"{base}.json")
        pred_path = get_pred_path(pipeline_name, fname)

        if not os.path.exists(gt_path):
            print(f"Missing GT for {base}, skipping")
            continue
        if not os.path.exists(pred_path):
            print(f"Missing predictions for {base}, skipping {pred_path}")
            continue
        
        id = f"{data_dir}/{fname}"
        
        gt_df = get_info(gt_path)
        gt_df['uid'] = id
        gt_df = gt_df[gt_df.element_type == target_element_type]
        pred_df = get_info(pred_path)
        pred_df['uid'] = id
        pred_df = pred_df[pred_df.element_type == target_element_type]

        if normalized_box:
            input_image = open_image(gt_path.replace("_gt.json",".png"))
            width, height = input_image.size
            denormalize_factor = width/1000
            pred_df['reconst_bounding_box'] = pred_df['bounding_box'].apply(lambda x: [int(i*denormalize_factor) for i in x])

        #Icons
        for consider_overlap in [True, False]:
            tp_df,fp_df,fn_df = get_matchings(pred_df, gt_df, consider_overlap=consider_overlap, normalized_box=normalized_box)
            tp, fp, fn, precision, recall = get_metrics(tp_df,fp_df,fn_df)
            for k,v in {'image':fname, 'tp':tp, 'fp':fp, 'fn':fn, 'precision':precision, 'recall':recall}.items():
                if consider_overlap not in result_dict:
                    result_dict[consider_overlap] = {}
                if k not in result_dict[consider_overlap]:
                    result_dict[consider_overlap][k] = []
                result_dict[consider_overlap][k].append(v)
            tp_dfs[consider_overlap].append(tp_df)
            fp_dfs[consider_overlap].append(fp_df)
            fn_dfs[consider_overlap].append(fn_df)
    
    for consider_overlap in [True, False]:
        result_df[consider_overlap] = pd.DataFrame(result_dict[consider_overlap])

    out_folder = f"03_evaluation/{target_element_type}/{pipeline_name}"
    os.makedirs(out_folder, exist_ok=True)
    out_path = f"{out_folder}/results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for consider_overlap in [True, False]:
            RES_DF = result_df[consider_overlap]
            TP_DF = pd.concat(tp_dfs[consider_overlap])
            FP_DF = pd.concat(fp_dfs[consider_overlap])
            FN_DF = pd.concat(fn_dfs[consider_overlap])
            suffix = ' no bbox' if not consider_overlap else ' with bbox'
            RES_DF.to_excel(writer, sheet_name=f"Result {suffix}", index=False)
            TP_DF.to_excel(writer, sheet_name=f"True Positives {suffix}", index=False)
            FP_DF.to_excel(writer, sheet_name=f"False Positives {suffix}", index=False)
            FN_DF.to_excel(writer, sheet_name=f"False Negatives {suffix}", index=False)
        summary_df = get_summary(result_df)
        summary_df.to_excel(writer, sheet_name=f"Result summary", index=False)
    
    for consider_overlap in [True, False]:
        FP_DF = pd.concat(fp_dfs[consider_overlap])
        FN_DF = pd.concat(fn_dfs[consider_overlap])
        suffix = ' no bbox' if not consider_overlap else ' with bbox'
        vis_df = fp_dfs[consider_overlap]
        for type_error in ['fp', 'fn']:
            vis_df = FP_DF if type_error == 'fp' else FN_DF
            out_folder_image = f"{out_folder}/sample_labels/{type_error} {suffix}"
            for ind, row in vis_df.iterrows():
                input_image_path = row['uid'].replace('_gt.json','.png')
                img = open_image(input_image_path)
                bbox = row["reconst_bounding_box"] if type_error == 'fp' else row["bounding_box"]
                label= row["represents"]
                bbox_str = "-".join([str(i) for i in bbox])
                img_cropped = img.crop(bbox)
                img_cropped.save(f"{out_folder_image}/{input_image_path.split('/')[-1].replace('.png','-' + label + '-' + bbox_str)}.png")

    if 'metric' not in SUPER_SUMMARY.columns:
        SUPER_SUMMARY['metric'] = summary_df['metric'].copy()
    summary_df[f'{pipeline}_{target_element}'] = summary_df['value']
    SUPER_SUMMARY = pd.concat([SUPER_SUMMARY, summary_df[f'{pipeline}_{target_element}']], axis = 0)
    print(f"Saved detailed results to {out_path}")
    print("Evaluation complete\n")


if __name__ == "__main__":
    DATA_DIRS = [p for p in glob("01_synthetic_data/*") if os.path.isdir(p)]
    PIPELINES = ["llm_gemma4-latest_icons","llm_gemma4-31b_icons","kde_scene_llm_gemma4-latest_icons","kde_scene_llm_gemma4-31b_icons"][::-1]
    SUPER_SUMMARY = pd.DataFrame()
    for target_element in ['Icon', 'Scene']:
        for pipeline in PIPELINES:
            print(f"evaluating pipeline {pipeline}")
            for data_dir in DATA_DIRS:
                evaluate(data_dir, pipeline_name=pipeline, normalized_box = True,
                target_element_type=target_element)
    SUPER_SUMMARY.to_excel('03_evaluation/super_summary.xlsx', index=False)

            
