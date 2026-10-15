import os
import json
from glob import glob
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from functools import lru_cache
from sentence_transformers import util
import math
from torch import tensor
import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
import json
import os
from PIL import Image, ImageDraw, ImageFont

@lru_cache(maxsize=300)
def _open_image_cached(path):
    return Image.open(path)

import torch
from torchvision.ops import box_iou

class CustomAREvaluator:
    def __init__(self, iou_thresholds=[0.30, 0.40, 0.50, 0.60, 0.70], max_dets=[30, 40, 50, 60, 70]):
        self.iou_thresholds = iou_thresholds
        self.max_dets = max_dets
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self):
        results = {}

        for K in self.max_dets:
            recalls = []

            for pred, tgt in zip(self.preds, self.targets):
                pred_boxes = pred["boxes"]
                pred_scores = pred["scores"]
                tgt_boxes = tgt["boxes"]

                if len(tgt_boxes) == 0:
                    recalls.append(1.0)
                    continue

                # Sort predictions by confidence
                idx = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[idx][:K]

                if len(pred_boxes) == 0:
                    recalls.append(0.0)
                    continue

                # IoU matrix
                ious = box_iou(pred_boxes, tgt_boxes)

                # Compute recall at each IoU threshold
                image_recalls = []
                for thr in self.iou_thresholds:
                    matched = (ious >= thr).any(dim=0).float()
                    recall = matched.sum().item() / len(tgt_boxes)
                    image_recalls.append(recall)

                # Average recall over IoU thresholds
                recalls.append(sum(image_recalls) / len(image_recalls))

            # Average recall over dataset
            results[f"mar_{K}"] = sum(recalls) / len(recalls)

        return results

class SemanticLabelMapper:
    def __init__(self, class_names, model_name="sentence-transformers/all-MiniLM-L6-v2", threshold=0.55):
        """
        class_names: list of canonical GT class names, e.g. ["remote", "laptop", "keyboard"]
        threshold: minimum similarity required to consider a match
        """
        self.class_names = class_names
        self.threshold = threshold

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Precompute embeddings for GT classes
        self.class_emb = self.model.encode(class_names, convert_to_tensor=True)

        # Build ID mapping
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}

    def map(self, predicted_label):
        """
        Returns integer class ID for a predicted text label.
        Uses semantic similarity to find the closest GT class.
        """
        pred_emb = self.model.encode(predicted_label, convert_to_tensor=True)

        # Compute cosine similarity
        sim = torch.nn.functional.cosine_similarity(pred_emb, self.class_emb)

        # Find best match
        best_idx = torch.argmax(sim).item()
        best_score = sim[best_idx].item()

        if best_score < self.threshold:
            # No good match → treat as unknown class
            return (-1, 0.0)

        return [best_idx, best_score]

    def map_batch(self, predicted_labels):
        """
        Map a list of predicted labels to integer IDs.
        """
        return [self.map(lbl) for lbl in predicted_labels]

class Evaluator:
    cache_export = {}
    semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def __init__(self, proximity_metric='iou', semantic_threshold=0.5, semantic_model_name="sentence-transformers/all-MiniLM-L6-v2", normalized_box=True, run_once=True):
        self.proximity_metric = proximity_metric
        proximity_threshold_dict = {'iou-10': 0.1,'iou-20': 0.2,'iou-30': 0.3,'iou-40': 0.4,'iou-50': 0.5,
                                    'iou-60': 0.6,'iou-70': 0.7,'iou-80': 0.8,'iou-90': 0.9,
                                    'iou': 0.5,'iou-min': 0.1,'iou-max': 0.9,
                                    'centerdist': 800,'centerdist-min': 0,'centerdist-max': 1000,'centerdist-threshold': 100,
                                    'centerdist-100': 100, 'centerdist-200':200, 'centerdist-400':400, 'centerdist-600':600, 'centerdist-800':800,
                                }
        self.proximity_threshold = proximity_threshold_dict[self.proximity_metric]
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.semantic_model_name = semantic_model_name
        self.semantic_threshold=semantic_threshold
        self.normalized_box = normalized_box
        self.run_once = run_once
        
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def _cached_similarity(text1, text2):
        emb1 = Evaluator.semantic_model.encode(text1, convert_to_tensor=True)
        emb2 = Evaluator.semantic_model.encode(text2, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()

        result = similarity

        # store raw inputs + output for export
        Evaluator.cache_export[(text1, text2)] = {
            "text1": text1,
            "text2": text2,
            "similarity": result,
        }
        return result
    
    def get_similarity_cache(self):
        df = pd.DataFrame(Evaluator.cache_export.values())
        df['Pipeline'] = self.pipeline
        return df


    def iou(self, boxA, boxB):
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
        ref_area = union
        return inter_area / ref_area if ref_area > 0 else 0.0

    @staticmethod
    def centerdist(boxA, boxB):
        # boxA and boxB are [x_min, y_min, x_max, y_max]
        ax = (boxA[0] + boxA[2]) / 2
        ay = (boxA[1] + boxA[3]) / 2
        bx = (boxB[0] + boxB[2]) / 2
        by = (boxB[1] + boxB[3]) / 2
        return math.sqrt((ax - bx)**2 + (ay - by)**2)

    # # Wrap only the expensive part: computing similarity
    # @lru_cache(maxsize=50000)
    # def _cached_similarity(self, text1, text2, threshold):
    #     emb1 = self.semantic_model.encode(text1, convert_to_tensor=True)
    #     emb2 = self.semantic_model.encode(text2, convert_to_tensor=True)
    #     similarity = util.cos_sim(emb1, emb2).item()
    #     return similarity >= threshold, similarity

    
    def open_image(self, path):
        return _open_image_cached(path)

    def are_semantically_similar(self, text1, text2, threshold=0.5):
        # text_ls = [text1.lower(), text2.lower()]
        # text_ls = sorted(text_ls)
        # text1, text2 = text_ls
        if text1 is None or text2 is None:
            return (False, 0.0)
        similarity = Evaluator._cached_similarity(text1, text2)
        return (similarity >= threshold, similarity)

    def enhance_sets(self, tup):
        set1, set2 = tup
        SET1 = set()
        for s1 in set1:
            similar_item = s1
            for s2 in set2:
                if self.are_semantically_similar(s1,s2):
                    similar_item = s2
                    break
            SET1.add(similar_item)
        return (SET1,set2)

    def get_info(self, element_path):
        elements = json.load(open(element_path))
        sample = [i for i in elements if i['element_type']=='Icon'][0]
        out_dict = {key:[] for key in sample.keys()}
        for element in elements:
            for key in out_dict.keys():
                info = element[key] if key in element else None
                out_dict[key].append(info)
        return pd.DataFrame(out_dict)

    def is_element_match(self, rowA, rowB, consider_proximity):
        """
        Returns True if:
        1. dictA['represents'] and dictB['represents'] are semantically similar
        2. IoU(dictA['bounding_box'], dictB['bounding_box']) >= iou_threshold
        """
        repA = rowA["represents"]
        repB = rowB["represents"]
        boxA = rowA["reconst_bounding_box"] if self.normalized_box else rowA['bounding_box']
        boxB = rowB["bounding_box"]

        if rowA['element_type'] != rowB['element_type']:
            return False #Because you cannot compare icons and scenes
        if rowA['element_type'] == 'Icon':
            semantically_similar, sem_score = self.are_semantically_similar(repA, repB, threshold=self.semantic_threshold)
            if not semantically_similar:
                return False
        if self.proximity_metric.startswith('iou'): 
            proximity = self.iou(boxA, boxB)
            if consider_proximity:
                if proximity < self.proximity_threshold:
                    return False
        elif self.proximity_metric.startswith('center'):
            proximity = Evaluator.centerdist(tuple(boxA), tuple(boxB))
            if consider_proximity:
                if proximity >= self.proximity_threshold:
                    return False
        return True

    # def is_element_match(self, rowA, rowB, consider_proximity):
    #     """
    #     Returns (is_match: bool, score: float).
    #     score is used to rank multiple valid candidates for the same
    #     prediction so we can pick the single best GT to match against.
    #     Higher score = better match.
    #     """
    #     repA = rowA["represents"]
    #     repB = rowB["represents"]
    #     boxA = rowA["reconst_bounding_box"] if self.normalized_box else rowA['bounding_box']
    #     boxB = rowB["bounding_box"]
    
    #     if rowA['element_type'] != rowB['element_type']:
    #         return False, 0.0  # cannot compare icons and scenes
    
    #     sem_score = 1.0  # default for element types that don't use semantic matching
    #     if rowA['element_type'] == 'Icon':
    #         sem_ok, sem_score = self.are_semantically_similar(
    #             repA, repB, threshold=self.semantic_threshold
    #         )
    #         if not sem_ok:
    #             return False, 0.0
    
    #     if not consider_proximity:
    #         # semantic-only matching, proximity not part of the gate or score
    #         return True, sem_score
    
    #     if self.proximity_metric.startswith('iou'):
    #         proximity = self.iou(boxA, boxB)
    #         if proximity < self.proximity_threshold:
    #             return False, 0.0
    #         # higher IoU + higher semantic similarity = better match
    #         score = 0.5 * sem_score + 0.5 * proximity
    
    #     elif self.proximity_metric.startswith('center'):
    #         proximity = Evaluator.centerdist(tuple(boxA), tuple(boxB))
    #         # FIX: threshold is a MAX allowed distance; smaller distance is
    #         # better. Reject when the boxes are too far apart, not too close.
    #         if proximity > self.proximity_threshold:
    #             return False, 0.0
    #         # convert distance to a 0..1 "closeness" score so it combines
    #         # sensibly with sem_score for ranking candidates
    #         closeness = 1.0 - (proximity / self.proximity_threshold)
    #         score = 0.5 * sem_score + 0.5 * closeness
    
    #     else:
    #         raise ValueError(f"Unknown proximity_metric: {self.proximity_metric}")
    
    #     return True, score



    
    def get_pred_path(self, pipeline_name, fname):
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

    def get_matchings(self, pred_df, gt_df, consider_proximity):
        tp, fp, fn = [], [], []
        gt_matched = []
        gt_df['gt_represents'] = gt_df['represents'].copy()
        for pred_ind, pred_row in pred_df.iterrows():
            match_found = False
            for gt_ind, gt_row in gt_df.iterrows():
                if self.is_element_match(pred_row, gt_row, consider_proximity):
                    gt_row_filtered = gt_row.drop(labels=['represents'])
                    merged = pd.concat([pred_row, gt_row_filtered], axis=0)
                    merged = merged.to_frame().T  # make it a single-row DataFrame
                    if gt_ind not in gt_matched:
                        tp.append(merged)
                        match_found = True
                        gt_matched.append(gt_ind)
                        break
            if not match_found and pred_row['represents']:
                fp.append(pred_row.to_frame().T)
        for gt_ind, gt_row in gt_df.iterrows():
            if gt_ind not in gt_matched:
                fn.append(gt_row.to_frame().T)

        tp_df = pd.concat(tp, ignore_index=True) if len(tp)>0 else pd.DataFrame()
        fp_df = pd.concat(fp, ignore_index=True) if len(fp)>0 else pd.DataFrame()
        fn_df = pd.concat(fn, ignore_index=True) if len(fn)>0 else pd.DataFrame()
        return tp_df, fp_df, fn_df


    def get_metrics(self, tp_df,fp_df,fn_df):
        tp = len(tp_df)
        fp = len(fp_df)
        fn = len(fn_df)
        try:
            precision = (tp / (tp + fp))
        except:
            precision = -1
        try:
            recall = (tp / (tp + fn))
        except: 
            recall = -1
        return tp, fp, fn, precision, recall

    def get_summary(self, result_df):
        summary_dict = {'metric':[], 'value':[]}
        for key, df in result_df.items():
            prefix = "not " if not key else ""

            for metric in ['precision', 'recall']:
                summary_dict["metric"].append(f"{prefix}considering bbox mean. {metric}")
                summary_dict["value"].append(df[metric].mean())
            #----
            summary_dict["metric"].append(f"{prefix}considering bbox dataset precision") 
            try:
                summary_dict["value"].append(df['tp'].sum()/(df['tp'].sum()+df['fp'].sum()))
            except:
                summary_dict["value"].append(-1)
            #----
            summary_dict["metric"].append(f"{prefix}considering bbox dataset recall")
            try:
                summary_dict["value"].append(df['tp'].sum()/(df['tp'].sum()+df['fn'].sum()))
            except:
                summary_dict["value"].append(-1)

            # summary_dict["metric"].append("")
            # summary_dict["value"].append(len(df['']))
        
        summary_df = pd.DataFrame(summary_dict)
        return summary_df

    def visualize_image(self, image_path, gt_df, tp_df, fp_df, fn_df, save_path):
        """
        Draws:
        - GT boxes in YELLOW
        - TP boxes in GREEN
        - FP boxes in RED
        - FN boxes in ORANGE
        """

        img = self.open_image(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Colors
        COLOR_GT = (255, 255, 0)     # yellow
        COLOR_TP = (0, 255, 0)       # green
        COLOR_FP = (255, 0, 0)       # red
        COLOR_FN = (255, 165, 0)     # orange
        # Helper to draw boxes
        def draw_box(row, color, tag):
            box = row["reconst_bounding_box"] if "reconst_bounding_box" in row else row["bounding_box"]
            x1, y1, x2, y2 = box

            # FN gets thicker line
            line_width = 3
            # 5 if tag == "FN" else 3
            offset = 10 if (tag in ("TP","FP")) else 0

            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            draw.text((x1, y1+offset), f"{tag}: {row['represents']}", fill=color)

        # Draw GT
        for _, row in gt_df.iterrows():
            draw_box(row, COLOR_GT, "GT")

        # Draw TP
        for _, row in tp_df.iterrows():
            draw_box(row, COLOR_TP, "TP")

        # Draw FP
        for _, row in fp_df.iterrows():
            draw_box(row, COLOR_FP, "FP")

        # Draw FN
        for _, row in fn_df.iterrows():
            draw_box(row, COLOR_FN, "FN")

        img.save(save_path)


    def evaluate(self, data_dir, pipeline_name, ref_path, target_element_type = 'Icon', normalized_box=False,):
        global SUPER_SUMMARY
        self.pipeline = pipeline_name
        self.out_mega_folder = f"03_evaluation/{pipeline_name}"
        result_dict = {}
        result_df = {}
        tp_dfs = {True:[], False:[]}
        fp_dfs = {True:[], False:[]}
        fn_dfs = {True:[], False:[]}
        GT_DF = pd.DataFrame()
        PRED_DF = pd.DataFrame()
        
        json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

        ref_files = glob(ref_path)
        json_ref_files = [os.path.basename(full_path).rstrip('_detected.json') for full_path in ref_files]

        json_files = [i for i in json_files if any([j in i for j in json_ref_files])]

        for fname in tqdm(json_files):
            if (not fname.endswith(".json")) or ("data_summary" in fname):
                continue
            # print(f"evaluating {fname}")
            base = fname.replace(".json", "")
            gt_path = os.path.join(data_dir, f"{base}.json")
            pred_path = self.get_pred_path(pipeline_name, fname)

            if not os.path.exists(gt_path):
                print(f"Missing GT for {base}, skipping")
                continue
            if not os.path.exists(pred_path):
                print(f"Missing predictions for {base}, skipping {pred_path}")
                continue
            
            id = f"{data_dir}/{fname}"
            
            gt_df = self.get_info(gt_path)
            gt_df['uid'] = id
            gt_df = gt_df[gt_df.element_type == target_element_type]
            if len(GT_DF) == 0:
                GT_DF = gt_df
            else:
                GT_DF = pd.concat([GT_DF,gt_df])

            pred_df = self.get_info(pred_path)
            pred_df['uid'] = id
            pred_df = pred_df[pred_df.element_type == target_element_type]

            if normalized_box:
                input_image = self.open_image(gt_path.replace("_gt.json",".png"))
                width, height = input_image.size
                denormalize_factor = width/1000
                pred_df['reconst_bounding_box'] = pred_df['bounding_box'].apply(lambda x: [int(i*denormalize_factor) for i in x])

            if len(PRED_DF) == 0:
                PRED_DF = pred_df
            else:
                PRED_DF = pd.concat([PRED_DF,pred_df])
            

            #Icons
            for consider_proximity in [True, False]:
                tp_df,fp_df,fn_df = self.get_matchings(pred_df, gt_df, consider_proximity=consider_proximity)

                vis_out_folder = f"{self.out_mega_folder}/annotated_images/{'with_bbox' if consider_proximity else 'no_bbox'}"
                os.makedirs(vis_out_folder, exist_ok=True)
                image_path = gt_path.replace("_gt.json", ".png")
                save_path = f"{vis_out_folder}/{fname.replace('.json', '')}.png"

                self.visualize_image(
                    image_path=image_path,
                    gt_df=gt_df,
                    tp_df=tp_df,
                    fp_df=fp_df,
                    fn_df=fn_df,
                    save_path=save_path
                )
                                
                tp, fp, fn, precision, recall = self.get_metrics(tp_df,fp_df,fn_df)
                for k,v in {'image':fname, 'tp':tp, 'fp':fp, 'fn':fn, 'precision':precision, 'recall':recall}.items():
                    if consider_proximity not in result_dict:
                        result_dict[consider_proximity] = {}
                    if k not in result_dict[consider_proximity]:
                        result_dict[consider_proximity][k] = []
                    result_dict[consider_proximity][k].append(v)
                tp_dfs[consider_proximity].append(tp_df)
                fp_dfs[consider_proximity].append(fp_df)
                fn_dfs[consider_proximity].append(fn_df)
        

        ##### calculating mAP-x

        gt_classes = GT_DF.represents.unique()
        mapper = SemanticLabelMapper(gt_classes,threshold=self.semantic_threshold, model_name=self.semantic_model_name)
        image_ids = GT_DF.uid.unique()
        preds = []
        target = []
        for image_id in image_ids:
            gt_rows = GT_DF[GT_DF.uid == image_id]
            pred_rows = PRED_DF[PRED_DF.uid==image_id]
            gt_ids_sims = mapper.map_batch(gt_rows['represents'])
            pred_ids_sims = mapper.map_batch(pred_rows['represents'])
            gt_ids = [k[0] for k in gt_ids_sims]

            pred_ids = [k[0] for k in pred_ids_sims]
            pred_sims = [k[1] for k in pred_ids_sims]
            pred_bboxes = pred_rows['reconst_bounding_box'].tolist()

            pred_ids_filt, pred_sims_filt, pred_bbox_filt = [],[],[]
            for (pi, ps, bb) in zip(pred_ids, pred_sims, pred_bboxes):
                if pi is not None:
                    pred_ids_filt.append(pi)
                    pred_sims_filt.append(ps)
                    pred_bbox_filt.append(bb)

            image_dict_pred = dict( boxes = tensor(pred_bbox_filt),
                                    scores = tensor(pred_sims_filt),
                                    labels = tensor(pred_ids_filt))
            image_dict_gt =  dict( boxes = tensor(gt_rows['bounding_box'].tolist()),
                                    labels = tensor(gt_ids))
            preds.append(image_dict_pred)
            target.append(image_dict_gt)
        map_metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", iou_thresholds=[0.30, 0.40, 0.50, 0.60, 0.70])
        map_metric.update(preds, target)
        result_map = map_metric.compute()
        result_map_df = pd.DataFrame({"metric":result_map.keys(), "value":[t.item() if (t.ndim == 0) else t.tolist() for t in result_map.values()]})

            


        #################################
        for consider_proximity in [True, False]:
            result_df[consider_proximity] = pd.DataFrame(result_dict[consider_proximity])
        
        out_folder = f"{self.out_mega_folder}/{self.proximity_metric}_sim-{self.semantic_threshold}/{target_element_type}"
        os.makedirs(out_folder, exist_ok=True)
        out_path = f"{out_folder}/results.xlsx"
        

        if self.run_once:
            with pd.ExcelWriter(f"{self.out_mega_folder}/Pred-GT.xlsx", engine="openpyxl") as writer:
                GT_DF.to_excel(writer, sheet_name=f"Gt", index=False)
                PRED_DF.to_excel(writer, sheet_name=f"Pred", index=False)
        
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for consider_proximity in [True, False]:
                RES_DF = result_df[consider_proximity]
                TP_DF = pd.concat(tp_dfs[consider_proximity])
                FP_DF = pd.concat(fp_dfs[consider_proximity])
                FN_DF = pd.concat(fn_dfs[consider_proximity])
                suffix = ' no bbox' if not consider_proximity else ' with bbox'
                
                TP_DF.to_excel(writer, sheet_name=f"True Positives {suffix}", index=False)
                FP_DF.to_excel(writer, sheet_name=f"False Positives {suffix}", index=False)
                FN_DF.to_excel(writer, sheet_name=f"False Negatives {suffix}", index=False)
                RES_DF.to_excel(writer, sheet_name=f"Result {suffix}", index=False)
            
            summary_df1 = self.get_summary(result_df)
            summary_df2 = result_map_df
            summary_df = pd.concat([summary_df1, summary_df2])
            summary_df.to_excel(writer, sheet_name=f"Result summary", index=False)
        print(f"Saved detailed results to {out_path}")
        
        for consider_proximity in [True, False]:
            FP_DF = pd.concat(fp_dfs[consider_proximity])
            FN_DF = pd.concat(fn_dfs[consider_proximity])
            suffix = ' no bbox' if not consider_proximity else ' with bbox'
            vis_df = fp_dfs[consider_proximity]
            for type_error in ['fp', 'fn']:
                vis_df = FP_DF if type_error == 'fp' else FN_DF
                out_folder_image = f"{out_folder}/sample_labels/{type_error} {suffix}"
                os.makedirs(out_folder_image, exist_ok=True)
                for ind, row in vis_df.iterrows():
                    input_image_path = row['uid'].replace('_gt.json','.png')
                    img = self.open_image(input_image_path)
                    bbox = row["reconst_bounding_box"] if type_error == 'fp' else row["bounding_box"]
                    label= row["represents"].replace('/','-slash-')
                    bbox_str = "-".join([str(i) for i in bbox])
                    img_cropped = img.crop(bbox)
                    # img_cropped.save(f"{out_folder_image}/{input_image_path.split('/')[-1].replace('.png','-' + label + '-' + bbox_str)}.png")
        similarity_cache = self.get_similarity_cache()
        
        return summary_df, similarity_cache


if __name__ == "__main__":
    TEST = True
    DATA_DIRS = [p for p in glob("01_synthetic_data/*") if os.path.isdir(p)]
    PIPELINES = ["kde_scene_kde_icon_clip_icons","llm_gemma4-latest_icons","llm_gemma4-31b_icons","kde_scene_llm_gemma4-latest_icons","kde_scene_llm_gemma4-31b_icons","kde_scene_llm_gemma4-31b_icons_ensemble_verify"]
    METRICS = ['iou-10','iou-20','iou-30','iou-40','iou-50','iou-60','iou-70','iou-80','iou-90','centerdist-100', 'centerdist-200', 'centerdist-400', 'centerdist-600', 'centerdist-800']
    SEM_THRESHS = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]
    SUPER_SUMMARY = pd.DataFrame()
    SIMILARITY_CACHE = None    
    REF_PATH = "/home/P103236/projects/paper_sketchobjectdetection/02_detected_picture_elements/kde_scene_llm_gemma4-31b_icons_ensemble_verify/02_to_05_clusters_05_to_20_classes_03_to_05_images_per_class_20_minimum_icons_00_to_00_percent_overlap_00_minimum_overlap_images_3.0_max_scaling_factor/image_*_detected.json"

    if TEST:
        PIPELINES = []#["kde_scene_kde_icon_clip_icons","llm_gemma4-latest_icons","llm_gemma4-31b_icons","kde_scene_llm_gemma4-latest_icons","kde_scene_llm_gemma4-31b_icons","kde_scene_llm_gemma4-31b_icons_ensemble_verify","kde_scene_llm_gemma4-31b_icons_ensemble_verify_w_majority","kde_scene_llm_gemma4-31b_icons_ensemble_verify_ignore_big"]
        for TARGET_MODEL in ["ministral-3:8b","llava:7b", "moondream:1.8b", "llama3.2-vision:latest", "minicpm-v:8b"]:
            PIPELINES.append(f"kde_scene_llm_gemma4-31b_icons_ensemble_verify_{TARGET_MODEL}_only")

        print(PIPELINES)
        # PIPELINES = ["kde_scene_llm_gemma4-31b_icons_ensemble_verify"]
        METRICS = ['iou-30']
        SEM_THRESHS = [0.375]
        SUPER_SUMMARY = pd.DataFrame()
        SIMILARITY_CACHE = None    

    for pipeline in PIPELINES:
        run_once = True
        for METRIC in METRICS:
            for SEM_THRESH in SEM_THRESHS:
                evaluator = Evaluator(proximity_metric=METRIC, semantic_threshold = SEM_THRESH, normalized_box=True, run_once=run_once)
                for target_element in ['Icon']:
                    print(f"evaluating pipeline {pipeline} | {METRIC} | {SEM_THRESH}")
                    for data_dir in DATA_DIRS:
                        summary_df, similarity_cache = evaluator.evaluate(data_dir, pipeline_name=pipeline, ref_path=REF_PATH, normalized_box = True, target_element_type=target_element)
                        if 'metric' not in SUPER_SUMMARY.columns:
                            SUPER_SUMMARY['metric'] = summary_df['metric'].copy().reset_index(drop=True)
                        if run_once:
                            if SIMILARITY_CACHE is None:
                                SIMILARITY_CACHE = similarity_cache
                            else:
                                SIMILARITY_CACHE = pd.concat([SIMILARITY_CACHE, similarity_cache])
                        row_index = f'{pipeline}_{target_element}|{METRIC}|{SEM_THRESH}'
                        summary_df[row_index] = summary_df['value']
                        SUPER_SUMMARY = pd.concat([SUPER_SUMMARY, summary_df[row_index].reset_index(drop=True)], axis = 1)
                        print("Evaluation complete\n")
                run_once = False
    SUPER_SUMMARY = SUPER_SUMMARY.set_index("metric").T.rename_axis("pipeline").reset_index()
    SUPER_SUMMARY.to_excel(f'03_evaluation/super_summary.xlsx', index=False)
    SIMILARITY_CACHE = SIMILARITY_CACHE.drop_duplicates(['text1','text2'])
    SIMILARITY_CACHE.to_excel(f'03_evaluation/similarity_cache.xlsx', index=False)
