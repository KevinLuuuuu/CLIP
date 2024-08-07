import json

with open("hw3_data/p2_data/val.json", newline='') as jsonfile:
    caption_data = json.load(jsonfile)
annotation_data = caption_data["annotations"]
images_id = caption_data["images"]

caption_json = {}
for item in images_id:
    caption_list = []
    id_caption_list = list(filter(lambda x:x["image_id"]==item["id"], annotation_data))
    for c in id_caption_list:
        caption_list.append(c["caption"])
    caption_json[item["file_name"]] = caption_list

json_object = json.dumps(caption_json, indent=2, ensure_ascii=False)
with open("caption_val.json", "w") as outfile:
    outfile.write(json_object)
print("caption.json ok !!")