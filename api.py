import sys
import json
from PIL import Image
import numpy as np
import traceback
from lang_sam import LangSAM

def run_langsam(image_path, text_prompt):
    try:
        model = LangSAM()
        image_pil = Image.open(image_path).convert("RGB")
        results = model.predict([image_pil], [text_prompt])
        
        original_image = Image.open(image_path).convert("RGBA")
        black_image = Image.new("RGBA", original_image.size, (0, 0, 0, 255))
        
        for result in results:
            for mask in result['masks']:
                binary_mask = (mask > 0.5).astype(np.uint8)
                result_image = Image.composite(original_image, black_image, Image.fromarray(binary_mask * 255))
                original_image = result_image
        
        output_image_path = "tmp/seg_out.png"
        result_image.save(output_image_path)
        return True
    
    except Exception as e:
        print(f"Error in run_langsam: {str(e)}")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read())
    image_path = input_data["image_path"]
    text_prompt = input_data["prompt"]
    result = run_langsam(image_path, text_prompt)
    