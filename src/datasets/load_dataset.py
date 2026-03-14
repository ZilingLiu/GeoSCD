import os
import json


def load_dataset(dataset_type, scene_dir, stride):
    if dataset_type == "changesim":
        json_path = os.path.join(scene_dir, f"pairs_{stride}.json")

        with open(json_path, 'r', encoding='utf-8') as f:
            img_pairs_json = json.load(f)
        img_pairs_list = img_pairs_json["pairs"]
        
        query_imgs_dir = os.path.join(scene_dir, "query")
        ref_imgs_dir = os.path.join(scene_dir, "ref")
        return query_imgs_dir, ref_imgs_dir, img_pairs_list
        
    elif dataset_type == "pscd":
        json_path = os.path.join(scene_dir, f"pairs_{stride}.json")

        with open(json_path, 'r', encoding='utf-8') as f:
            img_pairs_json = json.load(f)
        img_pairs_list = img_pairs_json["pairs"]
        
        query_imgs_dir = os.path.join(scene_dir, "t0")
        ref_imgs_dir = os.path.join(scene_dir, "t1")
        return query_imgs_dir, ref_imgs_dir, img_pairs_list
    elif dataset_type == "paslcd":
        json_path = os.path.join(scene_dir, f"pairs_{stride}.json")

        with open(json_path, 'r', encoding='utf-8') as f:
            img_pairs_json = json.load(f)
        img_pairs_list = img_pairs_json["pairs"]
        
        query_imgs_dir = os.path.join(scene_dir, "images")
        ref_imgs_dir = os.path.join(scene_dir, "images")
        return query_imgs_dir, ref_imgs_dir, img_pairs_list
        
        
                                                
    