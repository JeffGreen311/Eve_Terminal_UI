"""
Upload fixed LoRA files to HuggingFace
"""
from huggingface_hub import login, upload_file
import os

# Login with write token
login(token="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")

repo_id = "JeffGreen311/eve_cosmic_dreamscape_art_gen"
loras_dir = r"c:\Users\jesus\S0LF0RG3\S0LF0RG3_AI\Eve_Docker_Container\eve_loras"

emotions = ['joy', 'love', 'awe', 'sorrow', 'fear', 'rage', 'transcend']

print("🚀 Uploading fixed LoRA files to HuggingFace...")

for emotion in emotions:
    fixed_path = f"{loras_dir}\\{emotion}\\eve_{emotion}_lora_fixed.safetensors"
    if os.path.exists(fixed_path):
        print(f"⬆️  Uploading {emotion}_fixed...")
        upload_file(
            path_or_fileobj=fixed_path,
            path_in_repo=f"{emotion}_fixed.safetensors",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✅ {emotion}_fixed uploaded!")
    else:
        print(f"❌ Missing: {fixed_path}")

print("\n✅ All fixed LoRAs uploaded!")
print(f"🔗 View at: https://huggingface.co/{repo_id}")
print("\n📝 New URLs:")
for emotion in emotions:
    print(f'{emotion}: "https://huggingface.co/{repo_id}/resolve/main/{emotion}_fixed.safetensors"')
