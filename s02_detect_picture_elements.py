from classes.region_proposal import RegionProposer
from classes.object_detection import ObjectDetector, ObjectVerifier
from classes.pipeline import Pipeline
from glob import glob
import kagglehub, os
import time
berlin_path = kagglehub.dataset_download("zara2099/tu-berlin-hand-sketch-image-dataset")
berlin_path = os.path.join(berlin_path, "TUBerlin", "png_ready")
files = glob(f"{berlin_path}/*")
berlin_classes = [i.split('/')[-1] for i in files]


if __name__ == "__main__":
    
    TEST = False
    
    if TEST:
        IMAGE_DIRS = glob("test_data/*")
        PREDEFINED_LABELS = ["person","tree","car","calender", "cat", "pounds","house", "church"]
    else:
        IMAGE_DIRS = glob("01_synthetic_data/*")   
        PREDEFINED_LABELS = berlin_classes
    
    scene_proposal_dbscan = RegionProposer(method="dbscan", params={"epsilon":95, "min_sample":3}, name='scene')   
    scene_proposal_kde = RegionProposer(method="kde", params={"kde_thresh":0.3}, name='scene')  
    icon_proposal_dbscan = RegionProposer(method="dbscan", params={"epsilon":25, "min_sample":3}, name='object')   
    icon_proposal_kde = RegionProposer(method="kde", params={"kde_thresh":0.2}, name='icon')  
    object_detection_clip = ObjectDetector(method="clip", params={"predefined_labels":PREDEFINED_LABELS}, name='icons')
    object_detection_llm_big = ObjectDetector(method="llm", params={"model_name":"gemma4:31b"}, name='icons')
    object_detection_llm_small = ObjectDetector(method="llm", params={"model_name":"gemma4:latest"}, name='icons')
    # object_verification_ensemble_small = ObjectVerifier(models=["ministral-3:8b","llava:7b", "moondream:1.8b", "llama3.2-vision:latest", "minicpm-v:8b"], judge = "llama3.1:8b", method="ensemble", name="verify", use_known_label = True)
    object_verification_ensemble_small_ignore_big = ObjectVerifier(models=["ministral-3:8b","llava:7b", "llava:13b","gemma4:e4b", "moondream:1.8b", "llama3.2-vision:latest", "minicpm-v:8b"], judge = "llama3.1:8b", method="ensemble", name="verify_ignore_big", use_known_label = False)
    pipeline_small = Pipeline([object_detection_llm_small])
    pipeline_big = Pipeline([object_detection_llm_big])
    pipeline_scene_small = Pipeline([scene_proposal_kde, object_detection_llm_small])
    pipeline_scene_big = Pipeline([scene_proposal_kde, object_detection_llm_big])
    pipeline_clip = Pipeline([scene_proposal_kde, icon_proposal_kde, object_detection_clip])
    pipeline_scene_big_verify = Pipeline([scene_proposal_kde, object_detection_llm_big, object_verification_ensemble_small])
    pipeline_scene_big_verify_ignore_big = Pipeline([scene_proposal_kde, object_detection_llm_big, object_verification_ensemble_small_ignore_big])

    for selected_pipeline in [pipeline_scene_big_verify, pipeline_scene_big_verify_ignore_big, pipeline_small, pipeline_big, pipeline_scene_small, pipeline_scene_big, pipeline_clip]:
        IMAGE_DIRS = sorted(IMAGE_DIRS)
        for IMAGE_DIR in IMAGE_DIRS:
            if os.path.isdir(IMAGE_DIR):
                image_files = sorted(glob(os.path.join(IMAGE_DIR,"*.png")))
                for image_path in image_files:
                    if "summary" in image_path:
                        continue
                    print(f"Running pipeline on image: {image_path}")
                    status = selected_pipeline.run(image_path, skip_existing=True)
                    if status:
                        selected_pipeline.draw_result()
                    else:
                        print("Skipping this file already processed.")